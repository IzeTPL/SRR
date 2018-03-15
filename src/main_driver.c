#include<stdlib.h>
#include<stdio.h>
#include<math.h>

#include<omp.h>

#include"OpenCL_util.h"


// main driver for OpenCL example programs
int main(int argc, char** argv)
{
  
  int i;
  
  /*----------------INITIALIZATION PHASE----------------------*/

  // for all operations indicate explicit info messages
  int monitor = UTC_BASIC_INFO + 1;

#ifdef time_measurments
  t_begin = time_clock();
#endif

  // Create OpenCL contexts on all available platforms
  // contexts are stored in table with indices: 0-CPU, 1-GPU, 2-ACCELERATOR
  // table entry is NULL if there is no such context for a given platform
  int number_of_platforms = utr_ocl_create_contexts(UTC_OCL_ALL_PLATFORMS, monitor);

  if(number_of_platforms>1){
    printf("\nMore than one platform in a system! Check whether it's OK with the code and give the proper ID:\n");
    scanf("%d",  &i);
    utv_ocl_struct.current_platform_index = i;
  }
  else{
    utv_ocl_struct.current_platform_index = 0;
  }

#ifdef time_measurments
  t_end = time_clock();
  printf("EXECUTION TIME: creating contexts: %lf\n", t_end-t_begin);
#endif

#ifdef time_measurments
  t_begin = time_clock();
#endif

  int platform_index = utv_ocl_struct.current_platform_index;
  // create command queues on all devices 
  utr_ocl_create_command_queues(platform_index, 
				UTC_OCL_ALL_DEVICES, monitor);
  
#ifdef time_measurments
  t_end = time_clock();
  printf("EXECUTION TIME: creating command queues: %lf\n", t_end-t_begin);
#endif


  /*----------------KERNEL CREATION PHASE----------------------*/

  create_kernels();

  /*----------------EXECUTION PHASE----------------------*/

  execute_kernels();

  /*----------------FINALIZATION PHASE----------------------*/

  utr_ocl_cleanup();
  
  return 0;
}

