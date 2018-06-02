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
	__global int *result,
	__local int *tmpBuffer,
	const int n_items)
{
	uint gid = get_global_id(0);	//global ID
	uint lid = get_local_id(0);		//local ID
	uint sumHelper = 1;				//helper var for sum jumps


	//copy input to temp buffer
	//tmpBuffer[lid] = input[gid];

	// this is recommended... i don't really know why but it seems so be faster (ask prof)
	tmpBuffer[2 * lid] = input[2 * gid];
	tmpBuffer[2 * lid + 1] = input[2 * gid + 1];

	// GO UP
	// loop with half jumps
	for (uint s = n_items >> 1; s > 0; s >>= 1) 
	{
		barrier(CLK_LOCAL_MEM_FENCE); // wait for all threads so all use the same index multiplier

		if (lid < s) 
		{
			uint item1 = sumHelper * (2 * lid + 1) - 1;
			uint item2 = sumHelper * (2 * lid + 2) - 1;
			tmpBuffer[item2] += tmpBuffer[item1];
		}
		// double the sum index multipier
		sumHelper <<= 1;
	}

	// AND BACK DOWN

	// set last item to 0
	if (lid == 0) tmpBuffer[n_items - 1] = 0;

	// start at 1 and double the index with each step
	for (uint s = 1; s < n_items; s <<= 1) 
	{
		// half the sumHelper with each step
		sumHelper >>= 1;

		barrier(CLK_LOCAL_MEM_FENCE); // wait for all threads so all use the same index multipier

		if (lid < s) 
		{
			uint item1 = sumHelper * (2 * lid + 1) - 1;
			uint item2 = sumHelper * (2 * lid + 2) - 1;

			float t = tmpBuffer[item2];
			tmpBuffer[item2] += tmpBuffer[item1];
			tmpBuffer[item1] = t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE); // all must finish before we can write the result

	//copy result back
	//result[gid] = tmpBuffer[lid];

	//same situation as before... (ask prof)
	result[2 * gid] = tmpBuffer[2 * lid];
	result[2 * gid + 1] = tmpBuffer[2 * lid + 1];
}