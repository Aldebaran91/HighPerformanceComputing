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

	float xpos = cosTheta * ((float)ix - (float)W / 2) - sinTheta * ((float)iy - (float)H / 2) + W / 2;
	float ypos = sinTheta * ((float)ix - (float)W / 2) + cosTheta * ((float)iy - (float)H / 2) + H / 2;

	if ((int)xpos >= 0 && (int)xpos < W && (int)ypos >= 0 && (int)ypos < H)
	{
		dest_data[iy * W + ix] = src_data[(int)(floor(ypos * W + xpos))];
	}
}