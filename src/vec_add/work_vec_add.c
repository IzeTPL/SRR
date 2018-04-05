#include<stdlib.h>
#include<stdio.h>
#include <math.h>

#include <CL/cl.h>

#include"OpenCL_util.h"

#include"vec_add_host.h"

#include"system_util.h"


/**---------------------------------------------------------*/
int vec_add_host_2(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* C, 
  int N, 
  int Nr_threads, 
  int Work_group_size,
  const cl_context context, 
  const cl_kernel vec_add_kernel, 
  const cl_command_queue queue
		) 
{ 

  int i, retval;

  printf("\n\n*****------ Starting execution of vec_add for size %d (%d MB) ------*****\n",
	 N, N*sizeof(SCALAR)/1024/1024);
  printf("*****---------------- Nr_threads %d, Work_group_size %d ------------*****\n",
	 Nr_threads, Work_group_size);

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
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, NULL, NULL); 


  // Invoke kernel 
  retval = clSetKernelArg(vec_add_kernel, 0, sizeof(int), (void*)&N);
  retval |= clSetKernelArg(vec_add_kernel, 1, sizeof(cl_mem), (void*)&d_A);
  retval |= clSetKernelArg(vec_add_kernel, 2, sizeof(cl_mem), (void*)&d_B);
  retval |= clSetKernelArg(vec_add_kernel, 3, sizeof(cl_mem), (void*)&d_C);
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
  clEnqueueNDRangeKernel(queue, vec_add_kernel, work_dim, 0, 
			 globalWorkSize, localWorkSize, 0, 0, &ndrEvt); 
  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);
  double t2 = time_clock();
  // Calculate performance 
  cl_ulong startTime;
  cl_ulong endTime;
  
  // Get kernel profiling info 
  clGetEventProfilingInfo(ndrEvt,
			  CL_PROFILING_COMMAND_START,
			  sizeof(cl_ulong),
			  &startTime,
			  0);
  clGetEventProfilingInfo(ndrEvt,
			  CL_PROFILING_COMMAND_END,
			  sizeof(cl_ulong),
			  &endTime,
			  0);
  double time = (double)endTime - (double)startTime;
  printf("\nKernel execution internal: time %lf, GB/s = %lf\n", 
	 time*1.0e-9, 3*N*sizeof(SCALAR)/(time*1.0e-9)/1024/1024/1024);

  printf("Kernel execution external: time %lf,  GB/s = %lf\n\n", 
	 t2-t1, 3*N*sizeof(SCALAR)/(t2-t1)/1024/1024/1024);

  // to test...
  for(i=0;i<N;i++) C[i]=0.0;

  // Read C from device memory 
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_bytes, C, 0, 0, 0); 

  // to test...
  for(i=0;i<N;i++) {
    if(fabs(C[i]-1.0)>1.e-6){
      printf("Error for index %d: %.20lf + %.20lf = %.20lf != 1.0\n",
	     i, A[i], B[i], C[i]);
      getchar();
    }
  }
  // Free device memory 
  clReleaseMemObject(d_A); 
  clReleaseMemObject(d_B); 
  clReleaseMemObject(d_C); 

  
  return(0);

}

