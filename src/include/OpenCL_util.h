#ifndef OPENCL_LOCAL_H
#define OPENCL_LOCAL_H

#include<stdlib.h>
#include<stdio.h>

#include <CL/cl.h>

#define UTC_BASIC_INFO 2

#define UTC_OCL_MAX_NUM_KERNELS 10

#define UTC_OCL_ALL_PLATFORMS -1

#define UTC_OCL_ALL_DEVICES -1
#define UTC_OCL_DEVICE_CPU 0
#define UTC_OCL_DEVICE_GPU 1
#define UTC_OCL_DEVICE_ACCELERATOR 2

typedef struct {
  //  char name[128];
  cl_device_id id;
  int context_index;
  int utc_type;
  cl_device_type type;
  double global_mem_bytes; // in B
  double global_max_alloc; // in B
  double local_mem_bytes; // in B
  double constant_mem_bytes; // in B
  double cache_bytes; // in B
  int cache_line_bytes; // in B
  int max_num_comp_units;
  int max_work_group_size;
  cl_command_queue command_queue;
  int number_of_kernels;
  cl_program program[UTC_OCL_MAX_NUM_KERNELS];
  cl_kernel kernel[UTC_OCL_MAX_NUM_KERNELS];
} utt_ocl_device_struct;

typedef struct {
  //  char name[128];
  cl_platform_id id;
  // cl_uint number_of_devices;
  int number_of_devices;
  utt_ocl_device_struct *list_of_devices;
  cl_context list_of_contexts[3]; // always: [0]-CPU, [1]-GPU, [2]-accelerator
} utt_ocl_platform_struct;

typedef struct {
  //cl_uint preferred_alignment = 16;    
  //cl_uint number_of_platforms;
  int number_of_platforms;
  utt_ocl_platform_struct* list_of_platforms;
  int current_platform_index; 
} utt_ocl_struct;

utt_ocl_struct utv_ocl_struct;


extern void create_kernels();

extern void execute_kernels();

extern int utr_ocl_create_contexts(
  int Platform_id,
  int Monitor
  );

extern int utr_ocl_create_command_queues(
    int Platform_index,
    int Device_type,
    int Monitor
  );

extern int utr_ocl_create_kernel(
  int Platform_index,
  int Device_index,
  int Kernel_index,
  char* Kernel_name,
  const char* FileName,
  int Monitor
);

extern int utr_ocl_device_type(
  int Platform_index,
  int Device_index
);

extern cl_context utr_ocl_select_context(
  int Platform_index,
  int Device_index
);

extern int utr_ocl_CPU_context_exists(
  int Platform_index
);

extern int utr_ocl_GPU_context_exists(
  int Platform_index
);

extern cl_command_queue utr_ocl_select_command_queue(
  int Platform_index,
  int Device_index
);

extern cl_kernel utr_ocl_select_kernel(
  int Platform_index,
  int Device_index,
  int Kernel_index
);

extern double utr_ocl_calculate_execution_time(
  cl_event ndrEvt
				  );

extern void utr_ocl_cleanup();

#endif
