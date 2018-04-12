__kernel void mat_transp_1_kernel(
  __global float *a , 
  __global float *a_t,
  unsigned a_width, 
  unsigned a_height)
{

  int read_idx = get_global_id (0) + get_global_id (1) * a_width;
  int write_idx = get_global_id (1) + get_global_id (0) * a_height;

  a_t [ write_idx ] = a[ read_idx ];
}
