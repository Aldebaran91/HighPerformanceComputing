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
	__global int *a,
	__global int *r)
{
	__global int* b;
	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	uint dp = 1;

	b[gid] = a[gid];

	b[2 * lid] = a[2 * gid];
	b[2 * lid + 1] = a[2 * gid + 1];

	for (uint s = n_items >> 1; s > 0; s >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid < s) {
			uint i = dp * (2 * lid + 1) - 1;
			uint j = dp * (2 * lid + 2) - 1;
			b[j] += b[i];
		}

		dp <<= 1;
	}

	if (lid == 0) b[n_items - 1] = 0;

	for (uint s = 1; s < n_items; s <<= 1) {
		dp >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);

		if (lid < s) {
			uint i = dp * (2 * lid + 1) - 1;
			uint j = dp * (2 * lid + 2) - 1;

			float t = b[j];
			b[j] += b[i];
			b[i] = t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	r[2 * gid] = b[2 * lid];
	r[2 * gid + 1] = b[2 * lid + 1];
}