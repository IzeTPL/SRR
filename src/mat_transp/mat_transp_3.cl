#define PADDING 1
#define BLOCK_SIZE_X 32
#define STRIDE_Y (BLOCK_SIZE_X/4)
#define A_BLOCK_STRIDE (BLOCK_SIZE_X * a_width)
#define A_T_BLOCK_STRIDE (BLOCK_SIZE_X * a_height)

__kernel void mat_transp_3_kernel(
  __global float *a , 
  __global float *a_t,
  unsigned int a_width, 
  unsigned int a_height
)
{
  __local float a_local [(BLOCK_SIZE_X+PADDING)*BLOCK_SIZE_X];

      
      int base_idx_a =
      	get_group_id(0) * BLOCK_SIZE_X +
      	get_group_id(1) * A_BLOCK_STRIDE;
      
      int glob_idx_a =
	base_idx_a + get_local_id (0)
	+ a_width * get_local_id (1);

      a_local [ get_local_id (1)*(BLOCK_SIZE_X+PADDING) + get_local_id (0)] = a [glob_idx_a];
      a_local [ (get_local_id (1)+STRIDE_Y)*(BLOCK_SIZE_X+PADDING) + get_local_id (0)] = a [glob_idx_a+a_width*STRIDE_Y];
      a_local [ (get_local_id (1)+2*STRIDE_Y)*(BLOCK_SIZE_X+PADDING) + get_local_id (0)] = a [glob_idx_a+2*a_width*STRIDE_Y];
      a_local [ (get_local_id (1)+3*STRIDE_Y)*(BLOCK_SIZE_X+PADDING) + get_local_id (0)] = a [glob_idx_a+3*a_width*STRIDE_Y];

      int base_idx_a_t =
	get_group_id(1) * BLOCK_SIZE_X +
	get_group_id(0) * A_T_BLOCK_STRIDE;

      int glob_idx_a_t =
	base_idx_a_t + get_local_id (0)
	+ a_height * get_local_id (1);
      
      barrier (CLK_LOCAL_MEM_FENCE);
      a_t[glob_idx_a_t] = a_local[ get_local_id(0)*(BLOCK_SIZE_X+PADDING) + get_local_id(1)];
      a_t[glob_idx_a_t+a_height*STRIDE_Y] = a_local[ (get_local_id(0))*(BLOCK_SIZE_X+PADDING) + get_local_id(1)+STRIDE_Y];
      a_t[glob_idx_a_t+2*a_height*STRIDE_Y] = a_local[ (get_local_id(0))*(BLOCK_SIZE_X+PADDING) + get_local_id(1)+2*STRIDE_Y];
      a_t[glob_idx_a_t+3*a_height*STRIDE_Y] = a_local[ (get_local_id(0))*(BLOCK_SIZE_X+PADDING) + get_local_id(1)+3*STRIDE_Y];

}


