__kernel void scan_nvidia(
	__global int * restrict g_idata,
	__global int * g_odata,
	const int n)
{
	__global int* temp = g_idata;
	int thid = get_local_id(0);
	int pout = 1, pin = 0;
	printf("THID=%d, GLOBAL=%d \n", thid, get_global_id(0));

	temp[pout * n + thid] = (thid > 0) ? g_idata[thid - 1] : 0;

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int offset = 1; offset < n; offset *= 2)
	{
		pout = 1 - pout;
		pin = 1 - pout;

		temp[pout * n + thid] = temp[pin * n + thid];

		if (thid >= offset) {
			temp[pout * n + thid] += temp[pin * n + thid - offset];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	g_odata[thid] = temp[pout * n + thid];
}

__kernel void scan(
	__global int *input,
	__global int *result)
{
	__global int* tmpBuffer;
	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint n_items = get_global_size(0);
	uint dp = 1;

	tmpBuffer[gid] = input[gid];

	tmpBuffer[2 * lid] = input[2 * gid];
	tmpBuffer[2 * lid + 1] = input[2 * gid + 1];

	for (uint s = n_items >> 1; s > 0; s >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < s) {
			uint i = dp * (2 * lid + 1) - 1;
			uint j = dp * (2 * lid + 2) - 1;
			tmpBuffer[j] += tmpBuffer[i];
		}

		dp <<= 1;
	}

	if (lid == 0) tmpBuffer[n_items - 1] = 0;

	for (uint s = 1; s < n_items; s <<= 1) {
		dp >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < s) {
			uint i = dp * (2 * lid + 1) - 1;
			uint j = dp * (2 * lid + 2) - 1;

			float t = tmpBuffer[j];
			tmpBuffer[j] += tmpBuffer[i];
			tmpBuffer[i] = t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	result[2 * gid] = tmpBuffer[2 * lid];
	result[2 * gid + 1] = tmpBuffer[2 * lid + 1];
}