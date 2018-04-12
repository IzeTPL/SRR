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



#define BLOCK_SIZE 16
#define NR_GROUPS 16
#define MULT 16
#define WYMIAR (BLOCK_SIZE*NR_GROUPS*MULT)
#define ROZMIAR (WYMIAR*WYMIAR)
// Matrices are stored in row-major order: 
// M(row, col) = M( row * WYMIAR + col ) 

int verify_result(
  float* result,
  float* result_compare
		  )
{
  // Verify the result
  int result_OK = 1;
  int i,j;
  for(i = 0; i < WYMIAR; i++) {
    for(j = 0; j < WYMIAR; j++) {
      if(fabs(result[i*WYMIAR+j] - result_compare[i+j*WYMIAR])>1.e-6) {
	result_OK = 0;
	break;
      }
    }
  }
  printf("\t\t6. verifying results: ");
  if(result_OK) {
    printf("Output is correct\n");
  } else {
    printf("Output is incorrect\n");
    getchar();
    getchar();
    /* for(i = 0; i < WYMIAR; i++) { */
    /*   for(j = 0; j < WYMIAR; j++) { */
    /*   //for(i = 0; i < 10; i++) { */
    /* 	if(fabs(result[i*WYMIAR+j] - result_compare[i+j*WYMIAR])>1.e-9) { */
    /* 	  printf("%d %d %16.8f %16.8f\n",  */
    /* 	  	 i, j, result[i*WYMIAR+j], result_compare[i+WYMIAR*j]); */
    /* 	  getchar(); */
    /* 	} */
    /*   } */
    /* } */
    /* exit(0); */
  }
    /* for(i = 0; i < length; i++) { */
    /*   printf("%16.8f %16.8f\n", result[i], result_compare[i]); */
    /* } */

  return(result_OK);
}

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
				  "mat_transp_1_kernel", "mat_transp_1.cl", monitor);
  
#ifdef time_measurments
  t_end = time_clock();
  printf("EXECUTION TIME: creating CPU kernel %d: %lf\n", kernel_index, t_end-t_begin);
#endif
  
  // create the second kernel for GPU
  kernel_index = 1; 
  utr_ocl_create_kernel_dev_type( platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
				  // kernel name:         , file:
				  "mat_transp_2_kernel", "mat_transp_2.cl", monitor);
  
  // create the third kernel for GPU
  kernel_index = 2; 
  utr_ocl_create_kernel_dev_type( platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
				  // kernel name:         , file:
				  "mat_transp_3_kernel", "mat_transp_3.cl", monitor);
  
  }

}

  /*----------------EXECUTION PHASE----------------------*/
void execute_kernels()
{

  // for all operations indicate explicit info messages
  int monitor = UTC_BASIC_INFO + 1;

  // calculations are performed for the first platform (platform_index == 0)
  int platform_index = utv_ocl_struct.current_platform_index;
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[platform_index];

  int kernel_index;
  int i,j,n;

  double nr_access;
  double t1,t2;

  if(monitor>UTC_BASIC_INFO){
    printf("\n------------Starting execution phase----------------\n");
  }
  
  // create matrices
  n=WYMIAR;
  float* A = (float *) malloc(ROZMIAR*sizeof(float));
  float* B = (float *) malloc(ROZMIAR*sizeof(float));
  float* C = (float *) malloc(ROZMIAR*sizeof(float));
  
  for(i=0;i<ROZMIAR;i++) A[i]=1.0*i/1000000.0;
  
  nr_access= 2.0*ROZMIAR; // read + write
  
  printf("mat_transp: nr_access %lf\n", nr_access);
  
  // get hardware characteristics to select good matrix shape
  // the set of device characteristics stored in data structure
  int device_index = 0; 
  utt_ocl_device_struct device_struct = 
    utv_ocl_struct.list_of_platforms[platform_index].list_of_devices[device_index];
  double global_mem_bytes = device_struct.global_mem_bytes;
  double global_max_alloc = device_struct.global_max_alloc;
  double local_mem_bytes = device_struct.local_mem_bytes;
  double constant_mem_bytes = device_struct.constant_mem_bytes;
  int max_num_comp_units = device_struct.max_num_comp_units;
  int max_work_group_size = device_struct.max_work_group_size;
  
  // in a loop over devices (or for a selected device)
  int idev=0; 
  for(idev=0; idev<platform_struct.number_of_devices; idev++){
    
    // int device_type = .....
    // choose device_index
    // int device_index = utr_ocl_select_device(platform_index, device_type);
    int device_index = idev;
    int device_type = utr_ocl_device_type(platform_index, device_index); 
    
    if(device_index>0 && device_type==utr_ocl_device_type(platform_index, device_index-1)) break; 
    //if(device_type == UTC_OCL_DEVICE_CPU) break;
    // choose the context
    cl_context context = utr_ocl_select_context(platform_index, device_index);  
    
    // choose the command queue
    cl_command_queue command_queue = 
      utr_ocl_select_command_queue(platform_index, device_index);  
    
    if(monitor>UTC_BASIC_INFO){
      printf("\nExecution: \t0. restoring context and command queue for platform %d and device %d\n",
	     platform_index, device_index);
    }
    
    if(context == NULL || command_queue == NULL){ 
      
      printf("failed to restore context and command queue for platform %d, device %d\n", 
	     platform_index, device_index);
      printf("%lu %lu\n", context, command_queue);
    }
    
    // choose the kernel

    if(device_type == UTC_OCL_DEVICE_GPU){
      
      for(kernel_index = 0; kernel_index<=2; kernel_index++){

	cl_kernel kernel = utr_ocl_select_kernel(platform_index, device_index, kernel_index);  
	
	if(monitor>UTC_BASIC_INFO){
	  printf("\n------------******************************************----------------\n");
	  printf("\nExecution: \t3. restoring kernel %d for platform %d and device %d\n",
		 kernel_index, platform_index, device_index);
	}
	
	if(context == NULL || command_queue == NULL || kernel == NULL){ 
	  
	  printf("failed to restore kernel for platform %d, device %d, kernel %d\n", 
		 platform_index, device_index, kernel_index);
	  printf("context %lu, command queue %lu, kernel %lu\n", 
		 context, command_queue, kernel);
	}
	
	for(i=0;i<ROZMIAR;i++) B[i]=0.0;
	time_init(); t1 = time_clock();
	
	// call routine to perform matrix transposition
	mat_transp_host(kernel_index, A, B, n, context, kernel, command_queue);
	
	t2 = time_clock(); time_print();
	printf("GB/s = %lf\n\n", nr_access*sizeof(float)/(t2-t1)/1024/1024/1024);
	
	
	// verify result 
	verify_result(A, B);
	
      }

    }
    
  } // end loop over devices
  
  return;
}



// Matrix transposition - Host code 
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE 
int mat_transp_host(
  int kernel_index,
  float* A, 
  float* B, 
  int N, 
  const cl_context context, 
  const cl_kernel mat_transp_kernel, 
  const cl_command_queue queue
		) 
{ 

// Load A to device memory 
  size_t size_bytes = N*N*sizeof(float); 
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
				size_bytes, NULL, NULL); 

  // Write A to device memory 
  clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size_bytes, A, 0, 0, 0); 

  // Allocate B in device memory 
  cl_mem d_B = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, NULL, NULL); 


  // Invoke kernel 
  clSetKernelArg(mat_transp_kernel, 0, sizeof(cl_mem), (void*)&d_A);
  clSetKernelArg(mat_transp_kernel, 1, sizeof(cl_mem), (void*)&d_B);
  clSetKernelArg(mat_transp_kernel, 2, sizeof(int), (void*)&N);
  clSetKernelArg(mat_transp_kernel, 3, sizeof(int), (void*)&N);

  size_t localWorkSize[3];
  size_t globalWorkSize[3];
  cl_uint work_dim;

  if(kernel_index==2){

    work_dim = 2;
    localWorkSize[0] = 32; 
    globalWorkSize[0] = N; 
    localWorkSize[1] = 32/4; 
    globalWorkSize[1] = N/4; 
    localWorkSize[2] = 0; 
    globalWorkSize[2] = 0; 


  }
  else{

    work_dim = 2;
    localWorkSize[0] = BLOCK_SIZE; 
    globalWorkSize[0] = N; 
    localWorkSize[1] = BLOCK_SIZE; 
    globalWorkSize[1] = N; 
    localWorkSize[2] = 0; 
    globalWorkSize[2] = 0; 
  
  }

  clFinish(queue);
  double t1 = time_clock();
  // Enqueue a kernel run call
  cl_event ndrEvt;
  clEnqueueNDRangeKernel(queue, mat_transp_kernel, work_dim, 0, 
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
	 time*1.0e-9, 2.0*N*N*sizeof(float)/(time*1.0e-9)/1024/1024/1024);

  printf("Kernel execution external: time %lf,  GB/s = %lf\n\n", 
	 t2-t1, 2.0*N*N*sizeof(float)/(t2-t1)/1024/1024/1024);

  // Read B from device memory 
  clEnqueueReadBuffer(queue, d_B, CL_TRUE, 0, size_bytes, B, 0, 0, 0); 

  // Free device memory 
  clReleaseMemObject(d_A); 
  clReleaseMemObject(d_B); 

  return(0);

}
