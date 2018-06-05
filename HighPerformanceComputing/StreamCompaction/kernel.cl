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
	__global int* output,
	uint bin_size
)
{
	const int gId = get_group_id(0);
	const int offset = gId * get_local_size(0);
	const int lid = get_local_id(0);
	output[lid + offset] = input[lid + offset] + sums[gId];

	printf("lid = %d / gId = %d / offset = %d / input = %d / sums = %d\n", lid, gId, offset, input[lid + offset], sums[gId]);

	/*int gid = get_global_id(0);
	int binId = get_group_id(0);*/

	//output[gid] = input[gid] + sums[binId];
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
	groupSums[binId] = output[group_offset + bin_size - 1];
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

	printf("lid = %d / offset = %d / mask = %d\n", lid, offset, mask[lid + offset]);

	if (mask[lid + offset] == 1)
	{
		printf("addr = %d / input = %d\n", addr[lid + offset], input[lid + offset]);
		output[addr[lid + offset]] = input[lid + offset];
	}
}