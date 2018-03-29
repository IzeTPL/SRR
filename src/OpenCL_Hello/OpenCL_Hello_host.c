#include<stdlib.h>
#include<stdio.h>
#include <math.h>

#include <CL/cl.h>

#include"OpenCL_util.h"

#define time_measurments

#ifdef time_measurments
#include"system_util.h"
  static double t_begin, t_end, t_total;
#endif

// for single or double precision calculations
// do not forget to choose the proper kernel!
//#define SCALAR double
#define SCALAR float

int verify_result(
		  int size,
		  SCALAR* result,
		  SCALAR* result_compare
		  );

int OpenCL_Hello_host(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* C, 
  int N, 
  const cl_context context, 
  const cl_kernel OpenCL_Hello_kernel, 
  const cl_command_queue queue
		      );

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
			); 

/*----------------KERNEL CREATION PHASE----------------------*/
void create_kernels()
{

  // for all operations indicate explicit info messages
  int monitor = UTC_BASIC_INFO + 1;

  int kernel_index;
  
  // calculations are performed for the selected platform
  int platform_index = utv_ocl_struct.current_platform_index;

  if(utr_ocl_GPU_context_exists(platform_index)){


#ifdef time_measurments
  t_begin = time_clock();
#endif

  // create the first kernel for GPU
  kernel_index = 0; 
  utr_ocl_create_kernel_dev_type( platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
				  // kernel name:         , file:
				  "OpenCL_Hello_kernel", "HelloWorld.cl", monitor);
  
#ifdef time_measurments
  t_end = time_clock();
  printf("EXECUTION TIME: creating CPU kernel %d: %lf\n", kernel_index, t_end-t_begin);
#endif
  
  }

}

/*----------------EXECUTION PHASE----------------------*/
void execute_kernels()
{
  
  // for all operations indicate explicit info messages
  int monitor = UTC_BASIC_INFO + 1;
  
  // calculations are performed for the selected platform 
  int platform_index = utv_ocl_struct.current_platform_index;
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[platform_index];
  
  int kernel_index;
  int i,j,k;
  
  if(monitor>UTC_BASIC_INFO){
    printf("\n------------Starting execution phase----------------\n");
  }
  
  // Option 1 - vector size as constant parameter
#define WORK_GROUP_SIZE 256
  //#define VECTOR_SIZE WORK_GROUP_SIZE
#define VECTOR_SIZE 64*1024*WORK_GROUP_SIZE
  
  // create arrays (vectors)
  int n=VECTOR_SIZE;
  
  // Option 2 - vector size read from terminal
  //printf("\nSet vector size:\n"); scanf("%d",  n);
  
  
  // create input and output arrays - in host memory
  
  SCALAR* A = (SCALAR *) malloc(n*sizeof(SCALAR));
  SCALAR* B = (SCALAR *) malloc(n*sizeof(SCALAR));
  SCALAR* SUM = (SCALAR *) malloc(n*sizeof(SCALAR));
  SCALAR* C = (SCALAR *) malloc(n*sizeof(SCALAR));
  
  for(i=0;i<n;i++) A[i]=(1.0*i)/(VECTOR_SIZE);
  for(i=0;i<n;i++) B[i]=1.0 - (1.0*i)/(VECTOR_SIZE);
  for(i=0;i<n;i++) {
    SUM[i] = A[i]+B[i];
  }

  // to test...
  for(i=0;i<n;i++) {
    if(fabs(SUM[i]-1.0)>1.e-6){
      printf("Error for index %d: %.20lf + %.20lf = %.20lf != 1.0\n",
	     i, A[i], B[i], SUM[i]);
      getchar();
    }
  }
  
  // in a loop over devices (or for a selected device)
  int idev=0; 
  for(idev=0; idev<platform_struct.number_of_devices; idev++){
    
    // int device_type = .....
    // choose device_index
    // int device_index = utr_ocl_select_device(platform_index, device_type);
    int device_index = idev;
    int device_type = utr_ocl_device_type(platform_index, device_index); 
    
    // to omit the second CPU or GPU
    if(device_index>0 && device_type==utr_ocl_device_type(platform_index, device_index-1)) break; 
    
    // to omit CPU
    if(device_type == UTC_OCL_DEVICE_CPU) break;
    
    utt_ocl_device_struct device_struct = 
      utv_ocl_struct.list_of_platforms[platform_index].list_of_devices[device_index];
    double global_mem_bytes = device_struct.global_mem_bytes;
    double global_max_alloc = device_struct.global_max_alloc;
    
    // choose the context
    cl_context context = utr_ocl_select_context(platform_index, device_index);  
    
    // choose the command queue
    cl_command_queue command_queue = 
      utr_ocl_select_command_queue(platform_index, device_index);  
    
    if(monitor>UTC_BASIC_INFO){
      printf("\nExecution: \t1. restoring context and command queue for platform %d and device %d\n",
	     platform_index, device_index);
    }
    
    if(context == NULL || command_queue == NULL){ 
      
      printf("failed to restore context and command queue for platform %d, device %d\n", 
	     platform_index, device_index);
      printf("%lu %lu\n", context, command_queue);
    }
    
    // choose the kernel
    kernel_index = 0;
    cl_kernel kernel = utr_ocl_select_kernel(platform_index, device_index, kernel_index);  
    
    if(monitor>UTC_BASIC_INFO){
      printf("\nExecution: \t2. restoring kernel %d for platform %d and device %d\n",
	     kernel_index, platform_index, device_index);
    }
    
    if(context == NULL || command_queue == NULL || kernel == NULL){ 
      
      printf("failed to restore kernel for platform %d, device %d, kernel %d\n", 
	     platform_index, device_index, kernel_index);
      printf("context %lu, command queue %lu, kernel %lu\n", 
	     context, command_queue, kernel);
    }
    
    int size_abc;
    // ---------- execution of single kernel with data manipulation

    for(i=0;i<n;i++) C[i]=0.0;
    time_init(); 
    
    size_abc = n;
    // call routine to perform the actual work....
    // pass OpenCL parameters and host input/output data
    OpenCL_Hello_host(kernel_index, A, B, C, size_abc, context, kernel, command_queue);
    
    time_print();
    
    
    // verify result 
    verify_result(n, SUM, C);
    
    // ---------- the end of: execution of single kernel with data manipulation
    
    // ---------- execution of single kernel with data manipulation

    for(i=0;i<n;i++) C[i]=0.0;
    time_init(); 
    
    size_abc = n;
    // call routine to perform the actual work....
    // pass OpenCL parameters and host input/output data
    FILE *f;
    f = fopen("wyniki.txt", "w");
    for(int i=0; i<n; i+=size_abc) {
    OpenCL_Hello_host_2(kernel_index, &A[i], &B[i], &C[i], size_abc, 
			WORK_GROUP_SIZE, context, kernel, command_queue, &f);
			
	}
    fclose(f);
    time_print();
    
    
    // verify result 
    verify_result(n, SUM, C);
    
    // ---------- the end of: execution of single kernel with data manipulation

  } // end loop over devices
  
  return;
}


/**---------------------------------------------------------*/
int OpenCL_Hello_host(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* C, 
  int N, 
  const cl_context context, 
  const cl_kernel OpenCL_Hello_kernel, 
  const cl_command_queue queue
		) 
{ 

  int retval;

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
  retval = clSetKernelArg(OpenCL_Hello_kernel, 0, sizeof(int), (void*)&N);
  retval |= clSetKernelArg(OpenCL_Hello_kernel, 1, sizeof(cl_mem), (void*)&d_A);
  retval |= clSetKernelArg(OpenCL_Hello_kernel, 2, sizeof(cl_mem), (void*)&d_B);
  retval |= clSetKernelArg(OpenCL_Hello_kernel, 3, sizeof(cl_mem), (void*)&d_C);
  if (retval != CL_SUCCESS) {
    printf("Failed to Set the kernel arguments.\n");
    exit(-1);
  }
  
  size_t globalWorkSize[3] = { N, 0, 0 };
  size_t localWorkSize[3] = { WORK_GROUP_SIZE, 0, 0 };
  cl_uint work_dim = 1;

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
  printf("\nKernel execution internal: time %lf, GB/s = %lf\n", 
	 time*1.0e-9, 3*N*sizeof(SCALAR)/(time*1.0e-9)/1024/1024/1024);

  printf("Kernel execution external: time %lf,  GB/s = %lf\n\n", 
	 t2-t1, 3*N*sizeof(SCALAR)/(t2-t1)/1024/1024/1024);

  // Read B from device memory 
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_bytes, C, 0, 0, 0); 

  // Free device memory 
  clReleaseMemObject(d_A); 
  clReleaseMemObject(d_B); 

  
  return(0);

}


/*---------------- Utilities ----------------------*/

int verify_result(
		  int size,
		  SCALAR* result,
		  SCALAR* result_compare
		  )
{
  // Verify the result
  int result_OK = 1;
  int i,j;
  for(i = 0; i < size; i++) {
    if(fabs(result[i] - result_compare[i])>1.e-6) {
      result_OK = 0;
      break;
    }
  }
  printf("\n\t\tVerifying results: ");
  if(result_OK) {
    printf("Output is correct\n");
  } else {
    printf("Output is incorrect\n");
    for(i = 0; i < size; i++) {
      printf("%16.8f %16.8f\n", result[i], result_compare[i]);
    }
    getchar();
    getchar();
  }
  
  return(result_OK);
}
