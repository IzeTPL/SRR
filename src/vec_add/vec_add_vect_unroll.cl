
__kernel void vec_add_kernel_vect_unroll(const int size,
					      __global const float4 *a,
					      __global const float4 *b,
					      __global float4 *c)
{
  int index_start = 4*get_global_id(0);
  int index_end = size/4;
  int stride = get_local_size(0) * get_num_groups(0) * 4;

  for (int i=index_start; i < index_end; i+=4*stride) {	
    c[i] = a[i]+b[i];
    c[i+stride] = a[i+stride]+b[i+stride];
    c[i+2*stride] = a[i+2*stride]+b[i+2*stride];
    c[i+3*stride] = a[i+3*stride]+b[i+3*stride];
  }
}
