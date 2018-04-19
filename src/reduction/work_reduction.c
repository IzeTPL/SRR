#include<stdlib.h>
#include<stdio.h>
#include <math.h>

#include <CL/cl.h>

#include"OpenCL_util.h"

#include"reduction_host.h"

#include"system_util.h"


/**---------------------------------------------------------*/
int reduction_host_2(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* sc_prod_p, 
  int N, 
  int Nr_threads, 
  int Work_group_size, 
  const cl_context context, 
  const cl_kernel reduction_kernel, 
  const cl_command_queue queue
		) 
{ 

  int i, retval;

  printf("\n\n*****------ Starting execution of reduction for size %d (%d MB) ------*****\n",
	 N, N*sizeof(SCALAR)/1024/1024);
  printf("*****---------------- Nr_threads %d, Work_group_size %d ------------*****\n",
	 Nr_threads, Work_group_size);

  double t_begin = time_clock(); // start time measurments - host

  int nr_work_groups=Nr_threads/Work_group_size;
// we create an array of size nr_work_groups to store work_group results
  SCALAR* C = (SCALAR *) malloc((nr_work_groups)*sizeof(SCALAR));
  size_t size_bytes_C = nr_work_groups*sizeof(SCALAR); 

  // Allocate A in device memory 
  size_t size_bytes = N*sizeof(SCALAR); 
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
				size_bytes, NULL, NULL); 

  // Write A to device memory 
  clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size_bytes, A, 0, 0, 0); 

  // Allocate B in device memory 
  cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,
				size_bytes, NULL, NULL); 

  // Write B to device memory 
  clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, size_bytes, B, 0, 0, 0); 

  // Allocate C in device memory 
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes_C, NULL, NULL); 


  // Invoke kernel 
  retval = clSetKernelArg(reduction_kernel, 0, sizeof(int), (void*)&N);
  retval |= clSetKernelArg(reduction_kernel, 1, sizeof(cl_mem), (void*)&d_A);
  retval |= clSetKernelArg(reduction_kernel, 2, sizeof(cl_mem), (void*)&d_B);
  retval |= clSetKernelArg(reduction_kernel, 3, sizeof(cl_mem), (void*)&d_C);
  if (retval != CL_SUCCESS) {
    printf("Failed to Set the kernel arguments.\n");
    exit(-1);
  }
  
  size_t globalWorkSize[3] = { Nr_threads, 0, 0 };
  size_t localWorkSize[3] = { Work_group_size, 0, 0 };
  cl_uint work_dim = 1;

  // wait for previous events to finish
  clFinish(queue);
  double t1 = time_clock(); // start time measurments - host
  // Enqueue a kernel run call
  cl_event ndrEvt;
  clEnqueueNDRangeKernel(queue, reduction_kernel, work_dim, 0, 
			 globalWorkSize, localWorkSize, 0, 0, &ndrEvt); 
  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);
  double t2 = time_clock();  

  // Calculate performance 
  double time_internal = utr_ocl_calculate_execution_time(ndrEvt);
  printf("\nKernel execution internal: time %lf, GB/s = %lf\n", 
	 time_internal, 3*N*sizeof(SCALAR)/(time_internal)/1024/1024/1024);

  printf("Kernel execution external: time %lf,  GB/s = %lf\n", 
	 t2-t1, 3*N*sizeof(SCALAR)/(t2-t1)/1024/1024/1024);

  // Read C from device memory 
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_bytes_C, C, 0, 0, 0); 

  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);

  // Free device memory 
  clReleaseMemObject(d_A); 
  clReleaseMemObject(d_B); 
  clReleaseMemObject(d_C); 

  printf("\n@@@@@@@@@ Total execution time: %lf @@@@@@@@@@@@\n\n", 
	 t2-t_begin);
  
  
  // calculate sc_prod
  SCALAR temp=0.0;
  for(i=0;i<nr_work_groups;i++) {
    temp+=C[i];
    //printf("work_group %d, sc_prod: local %f, total %f\n",
    //	   i, C[i], temp);
  }
  *sc_prod_p = temp;

  
  return(0);

}


