#include<stdlib.h>
#include<stdio.h>
#include <math.h>

#include <CL/cl.h>

#include"OpenCL_util.h"

#define time_measurments

#ifdef time_measurments
  #include"system_util.h"
#endif

// for single or double precision calculations
// do not forget to choose the proper kernel!
//#define SCALAR double
#define SCALAR float

/**---------------------------------------------------------*/
int OpenCL_Hello_host_2(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* C, 
  int N, 
  int work_group_size,
  const cl_context context, 
  const cl_kernel OpenCL_Hello_kernel, 
  const cl_command_queue queue,
  double *wyniki
		) 
{ 

  int retval;

	double time1;
	double time2;
	
	//time1=time_clock();
	//time2=time_clock();
	//clFinish(queue);
	//printf("Total time %lf", time2-time1);

  // Allocate A in device memory 
  time1=time_clock();
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
   clFinish(queue);
  time2=time_clock(); 
  
  //printf("\nTotal read time %lf\n", time2-time1);
  //printf("\nGB/s = %lf\n", 2*N*sizeof(SCALAR)/(time2-time1)/1024/1024/1024);

	wyniki[0] = time2-time1;
	wyniki[1] = 2*N*sizeof(SCALAR)/(time2-time1)/1024/1024/1024;

  // Allocate C in device memory 
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, NULL, NULL); 


  // Invoke kernel 
  retval = clSetKernelArg(OpenCL_Hello_kernel, 0, sizeof(int), (void*)&N);
  retval |= clSetKernelArg(OpenCL_Hello_kernel, 1, sizeof(cl_mem), (void*)&d_A);
  retval |= clSetKernelArg(OpenCL_Hello_kernel, 2, sizeof(cl_mem), (void*)&d_B);
  retval |= clSetKernelArg(OpenCL_Hello_kernel, 3, sizeof(cl_mem), (void*)&d_C);
  if (retval != CL_SUCCESS) {
    printf("Failed to Set the kernel arguments.\n");
    exit(-1);
  }
  
  size_t globalWorkSize[3] = { N, 0, 0 };
  size_t localWorkSize[3] = { work_group_size, 0, 0 };
  cl_uint work_dim = 1;

  A[0] = 2;
  B[0] = 2;

  // wait for previous events to finish
  clFinish(queue);
  double t1 = time_clock(); // start time measurments - host
  // Enqueue a kernel run call
  cl_event ndrEvt;
  clEnqueueNDRangeKernel(queue, OpenCL_Hello_kernel, work_dim, 0, 
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
  //printf("\nKernel execution internal: time %lf, GB/s = %lf\n",
	 //time*1.0e-9, 3*N*sizeof(SCALAR)/(time*1.0e-9)/1024/1024/1024);

	wyniki[2] = t2-t1;
	wyniki[3] = 3*N*sizeof(SCALAR)/(t2-t1)/1024/1024/1024;

  //printf("Kernel execution external: time %lf,  GB/s = %lf\n\n",
	 //t2-t1, 3*N*sizeof(SCALAR)/(t2-t1)/1024/1024/1024);

  // Read B from device memory 
  time1=time_clock();
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_bytes, C, 0, 0, 0); 
  clFinish(queue);
  time2=time_clock(); 
  //printf("\nTotal write time %lf", time2-time1);
  //printf("\nGB/s = %lf\n", 1*N*sizeof(SCALAR)/((time2-time1))/1024/1024/1024);

	wyniki[4] = time2-time1;
	wyniki[5] = 1*N*sizeof(SCALAR)/(time2-time1)/1024/1024/1024;

  // Free device memory 
  clReleaseMemObject(d_A); 
  clReleaseMemObject(d_B); 

  
  return(0);

}

