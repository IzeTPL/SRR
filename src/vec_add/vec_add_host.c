#include<stdlib.h>
#include<stdio.h>
#include <math.h>

#include <CL/cl.h>

#include"OpenCL_util.h"

#include"vec_add_host.h"

#define time_measurments

#ifdef time_measurments
#include"system_util.h"
  static double t_begin, t_end, t_total;
#endif


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
				  "vec_add_kernel", "vec_add.cl", monitor);
  
  // create the first kernel for GPU
  kernel_index = 1; 
  utr_ocl_create_kernel_dev_type( platform_index, UTC_OCL_DEVICE_GPU, kernel_index,
				  // kernel name:         , file:
				  "vec_add_kernel_vect_unroll", "vec_add_vect_unroll.cl", monitor);
  
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
  
#define WORK_GROUP_SIZE 128
#define MULT_MAX 128*1024
  //#define MULT_MAX 1

  // Option 1 - vector size as constant parameter
#define VECTOR_SIZE MULT_MAX*WORK_GROUP_SIZE
  
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


  printf("\n------------Starting OpenMP execution phase----------------\n");
  printf("\nArrays size: %d (%d MB)\n", VECTOR_SIZE, VECTOR_SIZE*sizeof(float)/1024/1024);
  t_begin = time_clock();

#pragma omp parallel for default(none) shared(n, SUM, A, B)
  for(i=0;i<n;i++) {
    SUM[i] = A[i]+B[i];
  }

#ifdef time_measurments
  t_end = time_clock();
  printf("EXECUTION TIME: executing standard loop: %lf\n", t_end-t_begin);
  printf("\tNumber of operations %d, performance %lf GFlops\n",
	 VECTOR_SIZE, VECTOR_SIZE / (t_end-t_begin) * 1e-9);
  printf("\tGBytes transferred to/from processor %lf, speed %lf GB/s\n",
	 3*VECTOR_SIZE*sizeof(float)*1e-9, 
	 3*VECTOR_SIZE*sizeof(float)/(t_end-t_begin)*1e-9);
#endif
  
  // to test...
  for(i=0;i<n;i++) {
    if(fabs(SUM[i]-1.0)>1.e-6){
      printf("Error for index %d: %.20lf + %.20lf = %.20lf != 1.0\n",
	     i, A[i], B[i], SUM[i]);
      getchar();
    }
  }
  
  printf("\n------------Starting OpenCL execution phase----------------\n");

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
    
    // choose the first kernel
    cl_kernel kernel;
    kernel_index = 0;
    kernel = utr_ocl_select_kernel(platform_index, device_index, kernel_index);  
    
    if(monitor>UTC_BASIC_INFO){
      printf("\n------------******************************************----------------\n");
      printf("\nExecution: \t2. restoring kernel %d for platform %d and device %d\n",
	     kernel_index, platform_index, device_index);
    }
    
    if(context == NULL || command_queue == NULL || kernel == NULL){ 
      
      printf("failed to restore kernel for platform %d, device %d, kernel %d\n", 
	     platform_index, device_index, kernel_index);
      printf("context %lu, command queue %lu, kernel %lu\n", 
	     context, command_queue, kernel);
    }
    

    // in a loop over vector sizes
    			FILE *f;
		f = fopen("wyniki.txt", "w");
    int mult;
    for(mult=1; mult<= MULT_MAX; mult*=2){
      
      // ---------- execution of single kernel with data manipulation
      printf("\n-------------W PĘTLI----------------\n");
          double *wyniki = (double*) malloc(6 * sizeof(double));

      
      for(i=0;i<n;i++) C[i]=0.0;

      time_init(); 
      
      int size_abc = n;
      //int nr_threads = n;
      int nr_threads = mult*WORK_GROUP_SIZE;
      // call routine to perform the actual work....
      // pass OpenCL parameters and host input/output data
      vec_add_host(kernel_index, A, B, C, size_abc, 
		   nr_threads, WORK_GROUP_SIZE, context, kernel, command_queue, wyniki);
		   fprintf(f, "%lf,%lf,%lf,%lf,%lf,%lf\n", wyniki[0], wyniki[1], wyniki[2], wyniki[3], wyniki[4], wyniki[5]);
      
      time_print();
      
      
      
      // verify result 
      verify_result( size_abc, SUM, C);
      
      // ---------- the end of: execution of single kernel with data manipulation
      
    }
    fclose(f);
    // choose the second kernel
    //kernel_index = 1;
    //kernel = utr_ocl_select_kernel(platform_index, device_index, kernel_index);  
    
    //if(monitor>UTC_BASIC_INFO){
    //  printf("\n------------******************************************----------------\n");
    //  printf("\nExecution: \t2. restoring kernel %d for platform %d and device %d\n",
//	     kernel_index, platform_index, device_index);
    //}
    
    //if(context == NULL || command_queue == NULL || kernel == NULL){ 
      
    //  printf("failed to restore kernel for platform %d, device %d, kernel %d\n", 
//	     platform_index, device_index, kernel_index);
    //  printf("context %lu, command queue %lu, kernel %lu\n", 
//	     context, command_queue, kernel);
    //}
    
    //for(mult=1; mult<= MULT_MAX/4/4/4; mult*=2){
    
      //// ---------- execution of single kernel with data manipulation
      
      //time_init(); 
      
      //int size_abc = n;
      //int nr_threads = mult*WORK_GROUP_SIZE;
      //// call routine to perform the actual work....
      //// pass OpenCL parameters and host input/output data
      //vec_add_host_2(kernel_index, A, B, C, size_abc, 
//		     nr_threads, WORK_GROUP_SIZE, context, kernel, command_queue);
      
      //time_print();
      
      
      //// verify result 
      //verify_result(size_abc, SUM, C);
      
      //// ---------- the end of: execution of single kernel with data manipulation
      
    //}
    
  } // end loop over devices
  
  return;
}


/**---------------------------------------------------------*/
int vec_add_host(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* C, 
  int N, 
  int Nr_threads, 
  int Work_group_size, 
  const cl_context context, 
  const cl_kernel vec_add_kernel, 
  const cl_command_queue queue,
  double *wyniki
		) 
{ 

  int i, retval;
  double t1,t2;
  cl_event ndrEvt;
  double time_internal;

  printf("\n\n*****------ Starting execution of vec_add for size %d (%d MB) ------*****\n",
	 N, N*sizeof(SCALAR)/1024/1024);
  printf("*****---------------- Nr_threads %d, Work_group_size %d ------------*****\n",
	 Nr_threads, Work_group_size);

  t1 = time_clock(); // start time measurments - host
  double t_begin = t1;
  double t_total = 0.0;

  // Allocate A in device memory 
  size_t size_bytes = N*sizeof(SCALAR); 
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY,
				size_bytes, NULL, NULL); 

  t2 = time_clock();
  printf("\nTime for creating buffer for A:  %lf\n", t2-t1);
  t_total += t2-t1;

  clFinish(queue);
  t1 = time_clock(); // start time measurments - host
  // Write A to device memory 
  clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, size_bytes, A, 0, 0,  &ndrEvt); 
  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);

  t2 = time_clock();  t_total += t2-t1;
  time_internal = utr_ocl_calculate_execution_time(ndrEvt);
  printf("\nTime for sending A to GPU memory:  %lf (internal %lf)\n", 
	 t2-t1, time_internal);
	 
	 wyniki[0] = t2-t1;
	 
  printf("\nBandwidth %lf GB/s (size %ul)\n", 
	 N*sizeof(SCALAR)/(time_internal)/1024/1024/1024, size_bytes);

  t1 = time_clock(); // start time measurments - host
  // Allocate B in device memory 
  cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY,
				size_bytes, NULL, NULL); 
  t2 = time_clock();  t_total += t2-t1;
  printf("\nTime for creating buffer for B:  %lf\n", t2-t1);

  clFinish(queue);
  t1 = time_clock(); // start time measurments - host

  // Write B to device memory 
  clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, size_bytes, B, 0, 0, &ndrEvt); 

  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);

  t2 = time_clock();  t_total += t2-t1;
  time_internal = utr_ocl_calculate_execution_time(ndrEvt);
  printf("\nTime for sending B to GPU memory:  %lf (internal %lf)\n", 
	 t2-t1, time_internal);
	 
	 wyniki[1] = t2-t1;
	 
  printf("\nBandwidth %lf GB/s (size %ul)\n", 
	 N*sizeof(SCALAR)/(time_internal)/1024/1024/1024, size_bytes);

  t1 = time_clock(); // start time measurments - host
  // Allocate C in device memory 
  cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size_bytes, NULL, NULL); 
  t2 = time_clock();  t_total += t2-t1;
  printf("\nTime for creating buffer for C:  %lf\n", t2-t1);


  // Invoke kernel 
  retval = clSetKernelArg(vec_add_kernel, 0, sizeof(int), (void*)&N);
  retval |= clSetKernelArg(vec_add_kernel, 1, sizeof(cl_mem), (void*)&d_A);
  retval |= clSetKernelArg(vec_add_kernel, 2, sizeof(cl_mem), (void*)&d_B);
  retval |= clSetKernelArg(vec_add_kernel, 3, sizeof(cl_mem), (void*)&d_C);
  retval |= clSetKernelArg(vec_add_kernel, 4, sizeof(int), (void*)&Nr_threads);
  if (retval != CL_SUCCESS) {
    printf("Failed to Set the kernel arguments.\n");
    exit(-1);
  }
  
  size_t globalWorkSize[3] = { Nr_threads, 0, 0 };
  size_t localWorkSize[3] = { Work_group_size, 0, 0 };
  cl_uint work_dim = 1;

  // wait for previous events to finish
  clFinish(queue);
  t1 = time_clock(); // start time measurments - host
  // Enqueue a kernel run call
  clEnqueueNDRangeKernel(queue, vec_add_kernel, work_dim, 0, 
			 globalWorkSize, localWorkSize, 0, 0, &ndrEvt); 
  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);
  t2 = time_clock();  t_total += t2-t1;

  // Calculate performance 
  time_internal = utr_ocl_calculate_execution_time(ndrEvt);
  printf("\nKernel execution internal: time %lf, GB/s = %lf\n", 
	 time_internal, 3*N*sizeof(SCALAR)/(time_internal)/1024/1024/1024);

  printf("Kernel execution external: time %lf,  GB/s = %lf\n", 
	 t2-t1, 3*N*sizeof(SCALAR)/(t2-t1)/1024/1024/1024);

	wyniki[2] = t2-t1;

  clFinish(queue);
  t1 = time_clock(); // start time measurments - host

  // to test...
  //for(i=0;i<N;i++) C[i]=0.0;

  // Read C from device memory 
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, size_bytes, C, 0, 0, &ndrEvt); 

  clWaitForEvents(1, &ndrEvt);
  clFinish(queue);

  t2 = time_clock();  t_total += t2-t1;
  time_internal = utr_ocl_calculate_execution_time(ndrEvt);
  printf("\nTime for sending C from GPU memory:  %lf (internal %lf)\n", 
	 t2-t1, time_internal);
	 
	 wyniki[3]= t2-t1;
	 
  printf("\nBandwidth %lf GB/s (size %ul)\n", 
	 N*sizeof(SCALAR)/(time_internal)/1024/1024/1024, size_bytes);

  // to test...
  //for(i=0;i<N;i++) {
  //  if(fabs(C[i]-1.0)>1.e-6){
  //    printf("Error for index %d: %.20lf + %.20lf = %.20lf != 1.0\n",
  //	     i, A[i], B[i], C[i]);
  //    getchar();
  //  }
  //}


  t1 = time_clock(); // start time measurments - host
  // Free device memory 
  clReleaseMemObject(d_A); 
  clReleaseMemObject(d_B); 
  clReleaseMemObject(d_C); 
  t2 = time_clock();  t_total += t2-t1;
  printf("\nTime for releasing memory buffers on GPU:  %lf\n", t2-t1);
  
  wyniki[4] = t2-t1;

  printf("\n@@@@@@@@@ Total execution time: %lf (%lf) @@@@@@@@@@@@\n\n", 
	 t2-t_begin, t_total);
  
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
      printf("i %d - %16.8f != %16.8f\n", i, result[i], result_compare[i]);
    getchar();
    getchar();
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
