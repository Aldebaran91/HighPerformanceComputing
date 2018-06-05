#define WARP_SHIFT 4
#define GRP_SHIFT 8
#define BANK_OFFSET(n) (((n) >> WARP_SHIFT) + ((n) >> GRP_SHIFT))

__kernel void predicateKernel_greater(
	__global const int* input, 
	__global int* output, 
	const int thresh)
{
	const int offset = get_group_id(0) * get_local_size(0);
	const int lid = get_local_id(0);
	output[lid + offset] = input[lid + offset] > thresh ? 1 : 0;
}

__kernel void ApplyGroupSums(
	__global const int* input,
	__global const int* sums,
	__global int* output
)
{
	int gid = get_global_id(0);
	int binId = get_group_id(0);

	output[gid] = input[gid] + sums[binId];
}

__kernel void blelloch(
	__global const int* input,
	__global int* output,
	__global int* groupSums,
	__local int* temp,
	uint bin_size
)
{
	int lid = get_local_id(0);
	int binId = get_group_id(0);
	int n = get_local_size(0) * 2;

	int group_offset = binId * bin_size;
	int maxval = 0;
	do
	{
		// calculate array indices and offsets to avoid SLM bank conflicts
		int ai = lid;
		int bi = lid + (n >> 1);
		int bankOffsetA = BANK_OFFSET(ai);
		int bankOffsetB = BANK_OFFSET(bi);

		// load input into local memory
		temp[ai + bankOffsetA] = input[group_offset + ai];
		temp[bi + bankOffsetB] = input[group_offset + bi];

		// parallel prefix sum up sweep phase
		int offset = 1;
		for (int d = n >> 1; d > 0; d >>= 1)
		{
			barrier(CLK_LOCAL_MEM_FENCE);
			if (lid < d)
			{
				int ai = offset * (2 * lid + 1) - 1;
				int bi = offset * (2 * lid + 2) - 1;
				ai += BANK_OFFSET(ai);
				bi += BANK_OFFSET(bi);
				temp[bi] += temp[ai];
			}
			offset <<= 1;
		}

		// clear the last element
		if (lid == 0)
		{
			groupSums[binId] = temp[n - 1 + BANK_OFFSET(n - 1)];
			temp[n - 1 + BANK_OFFSET(n - 1)] = 0;
		}

		// down sweep phase
		for (int d = 1; d < n; d <<= 1)
		{
			offset >>= 1;
			barrier(CLK_LOCAL_MEM_FENCE);

			if (lid < d)
			{
				int ai = offset * (2 * lid + 1) - 1;
				int bi = offset * (2 * lid + 2) - 1;
				ai += BANK_OFFSET(ai);
				bi += BANK_OFFSET(bi);

				int t = temp[ai];
				temp[ai] = temp[bi];
				temp[bi] += t;
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);

		//output scan result to global memory
		output[group_offset + ai] = temp[ai + bankOffsetA] + maxval;
		output[group_offset + bi] = temp[bi + bankOffsetB] + maxval;

		//printf("group_offset = %d / bankOffsetA = %d / bankOffsetB = %d / maxval = %d \n", group_offset, bankOffsetA, bankOffsetB, maxval);

		//update cumulative prefix sum shift and histogram index for next iteration
		maxval += temp[n - 1 + BANK_OFFSET(n - 1)] + input[group_offset + n - 1];
		group_offset += n;
	} while (group_offset < (binId + 1) * bin_size);


	
}

// blellock without BC avoidance... somehow messes up my arrays
// better it works slower then not at all...
__kernel void blelloch_simple(
	__global const int* input,
	__global int* output,
	__global int* groupSums,
	__local int* temp,
	uint bin_size
)
{
	int gid = get_group_id(0);
	int lid = get_local_id(0);

	// sizes & offset
	int size_local = get_local_size(0);
	int size_global = get_global_size(0);
	int group_offset = gid * size_local;

	// again... not sure why but everyone does it
	temp[lid] = input[lid + group_offset];
	temp[lid + 1] = input[lid + group_offset + 1];

	//upsweep
	int offset = 1;
	for (int d = size_local >> 1; d > 0; d >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		if (lid < d)
		{
			int ai = offset * (2 * lid + 1) - 1;
			int bi = offset * (2 * lid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset <<= 1;
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// save blocksum & clear the last element
	if (lid == 0)
	{
		groupSums[gid] = temp[size_local - 1];
		temp[size_local - 1] = 0;
	}

	//downsweep
	for (int d = 1; d < size_local; d <<= 1)
	{
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

		if (lid < d)
		{
			int ai = offset * (2 * lid + 1) - 1;
			int bi = offset * (2 * lid + 2) - 1;
			int t = temp[ai];

			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	// same as up there but with writing out
	output[group_offset + lid] = temp[lid];
	output[group_offset + lid + 1] = temp[lid + 1];
}

__kernel void scatter(
	__global const int* restrict input,
	__global const int* restrict addr,
	__global const int* restrict mask,
	__global int* output
)
{
	const int offset = get_group_id(0) * get_local_size(0);
	const int lid = get_local_id(0);

	if (mask[lid + offset] == 1)
	{
		output[addr[lid + offset]] = input[lid + offset];
	}
}

