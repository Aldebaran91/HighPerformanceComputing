__kernel void image_rotate(
	__global float * src_data,
	__global float * dest_data,
	float sinTheta,
	float cosTheta)
{
	const int ix = get_global_id(0);
	const int iy = get_global_id(1);
	int W = get_global_size(0);
	int H = get_global_size(1);
	int dest = W * iy + ix;
	int w2 = W / 2;
	int h2 = H / 2;
	
	int xpos = (int)floor((cosTheta * (ix - w2))
		- (sinTheta * (iy - h2)) + w2);
	int ypos = (int)floor((sinTheta * (ix - w2))
		+ (cosTheta * (iy - h2)) + h2);
	int pos = W * ypos + xpos;

	if (xpos >= 0 && xpos < W 
		&& ypos >= 0 && ypos < H
		&& pos >= 0 && pos < (W * H))
	{
		dest_data[dest] = src_data[pos];
	}
}