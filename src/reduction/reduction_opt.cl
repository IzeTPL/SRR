#define SCALAR double


__kernel void reduction_opt_kernel(const int size,
			   __global const SCALAR *a,
		           __global const SCALAR *b,
			   __global SCALAR *c)
{

  int index_start = get_global_id(0);
  int index_end = size;
  int stride = get_global_size(0);


  __local SCALAR sc_prod_local[256];

  SCALAR temp = 0.0;
  for (int i=index_start; i < index_end; i+=stride) {	
    temp += a[i]*b[i];
  }

  sc_prod_local[get_local_id(0)] = temp;

  barrier(CLK_LOCAL_MEM_FENCE); 

  // reduction phase
  if(get_local_id(0)==0){
    temp=0.0;
    for(int i=0; i<get_local_size(0); i++){
      temp += sc_prod_local[i];
    }
    c[get_group_id(0)] = temp;
  }

}
