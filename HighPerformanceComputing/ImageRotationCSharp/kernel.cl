
/*
* a kernel that add the elements of two vectors pairwise
*/
__kernel void vector_add(
	__global const int *A,
	__global const int *B,
	__global int *C)
{
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}

/*
*
*/
__kernel void image_rotate(
	__global float * src_data,
	__global float * dest_data,
	int W, int H,
	float sinTheta, float cosTheta)
{
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);

	float xpos = (((float)ix)*cosTheta + ((float)iy)*sinTheta);
	float ypos = (((float)iy)*cosTheta - ((float)ix)*sinTheta);

	if ((int)xpos >= 0 && (int)xpos < W && (int)ypos >= 0 && (int)ypos < H)
	{
		dest_data[iy*W + ix] = src_data[(int)(floor(ypos*W + xpos))];
	}
}