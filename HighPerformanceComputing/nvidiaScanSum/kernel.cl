__kernel void scan(
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