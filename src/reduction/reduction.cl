
__kernel void reduction_kernel(const int size,
			   __global const float *a,
		           __global const float *b,
			   __global float *c)
{

  int index_start = get_global_id(0);
  int index_end = size;
  int stride = get_global_size(0);

  __local float sc_prod_local[256];

  float temp = 0.0;
  for (int i=index_start; i < index_end/4; i+=4*stride) {	
    temp += a[i]*b[i];
    temp += a[i+stride]*b[i+stride];
    temp += a[i+2*stride]*b[i+2*stride];
    temp += a[i+3*stride]*b[i+3*stride];
  }

  sc_prod_local[get_local_id(0)] = temp;

  if(get_local_id(0)==0){
    temp=0.0;
    for(int i=0; i<get_local_size(0); i++){
      temp += sc_prod_local[i];
    }
    c[get_group_id(0)] = temp;
  }

}
