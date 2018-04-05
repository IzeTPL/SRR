
__kernel void vec_add_kernel(const int size,
			   __global const float *a,
		           __global const float *b,
			   __global float *c,
			   const int threads
)
{

	//blokowo

    //int gid = get_global_id(0);
    //int size_per_thread = size/threads;
  //int index_start = gid*size_per_thread;
  //int index_end = (gid+1) * size_per_thread;

  //for (int i=index_start; i < index_end && i < size; i++) {	
		//c[i] = a[i] + b[i];
  //}
  
  //cyklicznie
  
	int index_start = get_global_id(0);
	int index_end = size;
	int stride = threads;
	
	for (int i=index_start; i < index_end; i+=stride) {
		//c[i] = a[i]+b[i];
	}
    
//if(gid<size) c[gid] = a[gid] + b[gid];

}
