
__kernel void OpenCL_Hello_kernel(const int size,
			   __global const float *a,
		           __global const float *b,
			   __global float *c)
{
    int gid = get_global_id(0);

    if(gid<size) c[gid] = a[gid] + b[gid];
}
