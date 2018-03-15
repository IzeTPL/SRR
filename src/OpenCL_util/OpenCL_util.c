#include <CL/cl.h>

#include"OpenCL_util.h"

void DisplayPlatformInfo(
			 cl_platform_id id, 
			 cl_platform_info name,
			 char* str)
{
  cl_int retval;
  size_t paramValueSize;
  
  retval = clGetPlatformInfo(
			     id,
			     name,
			     0,
			     NULL,
			     &paramValueSize);
  if (retval != CL_SUCCESS){
    printf("Failed to find OpenCL platform %s.\n", str);
    return;
  }
  
  char * info = (char *)malloc(sizeof(char) * paramValueSize);
  retval = clGetPlatformInfo(
			     id,
			     name,
			     paramValueSize,
			     info,
			     NULL);
  if (retval != CL_SUCCESS)  {
    printf("Failed to find OpenCL platform %s.\n", str);
    return;
  }
  
  printf("\t%s:\t%s\n", str, info );
  free(info); 
}

void DisplayDeviceInfo(
		       cl_device_id id, 
		       cl_device_info name,
		       char* str)
{
  cl_int retval;
  size_t paramValueSize;
  
  retval = clGetDeviceInfo(
			   id,
			   name,
			   0,
			   NULL,
			   &paramValueSize);
  if (retval != CL_SUCCESS) {
    printf("Failed to find OpenCL device info %s.\n", str);
    return;
  }
  
  char * info = (char *)malloc(sizeof(char) * paramValueSize);
  retval = clGetDeviceInfo(
			   id,
			   name,
			   paramValueSize,
			   info,
			   NULL);
  
  if (retval != CL_SUCCESS) {
    printf("Failed to find OpenCL device info %s.\n", str);
    return;
  }

  printf("\t\t%s:\t%s\n", str, info );
  free(info);
};

///
//  Create OpenCL contexts
//
int utr_ocl_create_contexts(
  int Chosen_platform_id,
  int Monitor
  )
{
  cl_int retval;
  cl_uint numPlatforms;
  cl_platform_id * platformIds;
  cl_context context = NULL;
  cl_uint i,j,k;
  
  // First, query the total number of platforms
  retval = clGetPlatformIDs(0, (cl_platform_id *) NULL, &numPlatforms);

  // allocate memory for local platform structures
  utv_ocl_struct.number_of_platforms = numPlatforms;
  utv_ocl_struct.list_of_platforms = 
    (utt_ocl_platform_struct *) malloc( sizeof(utt_ocl_platform_struct)
					* numPlatforms);

  // Next, allocate memory for the installed platforms, and qeury 
  // to get the list.
  platformIds = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);
  retval = clGetPlatformIDs(numPlatforms, platformIds, NULL);

  if(Monitor>=UTC_BASIC_INFO){
    printf("\nNumber of OpenCL platforms: \t%d\n", numPlatforms); 
  }

  // Iterate through the list of platforms displaying associated information
  for (i = 0; i < numPlatforms; i++) {

    if(Monitor>UTC_BASIC_INFO){
      printf("\n");
      printf("Platform %d:\n", i); 
    }

    utv_ocl_struct.list_of_platforms[i].id = platformIds[i];
    //clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, size_of_name???, 
    //		      utv_ocl_struct.list_of_platforms[i].name, (size_t *) NULL);

    if(Monitor>UTC_BASIC_INFO){

      // First we display information associated with the platform
      DisplayPlatformInfo(
			platformIds[i], 
			CL_PLATFORM_NAME, 
			"CL_PLATFORM_NAME");
      DisplayPlatformInfo(
			platformIds[i], 
			CL_PLATFORM_PROFILE, 
			"CL_PLATFORM_PROFILE");
      DisplayPlatformInfo(
			platformIds[i], 
			CL_PLATFORM_VERSION, 
			"CL_PLATFORM_VERSION");
      DisplayPlatformInfo(
			platformIds[i], 
			CL_PLATFORM_VENDOR, 
			"CL_PLATFORM_VENDOR");
    }

    // Now query the set of devices associated with the platform
    cl_uint numDevices;
    retval = clGetDeviceIDs(
			    platformIds[i],
			    CL_DEVICE_TYPE_ALL,
			    0,
			    NULL,
			    &numDevices);


    utv_ocl_struct.list_of_platforms[i].number_of_devices = numDevices;
    utv_ocl_struct.list_of_platforms[i].list_of_devices = 
      (utt_ocl_device_struct *) malloc( sizeof(utt_ocl_device_struct) 
					* numDevices);

    cl_device_id * devices = 
      (cl_device_id *) malloc (sizeof(cl_device_id) * numDevices);

    retval = clGetDeviceIDs(
			    platformIds[i],
			    CL_DEVICE_TYPE_ALL,
			    numDevices,
			    devices,
			    NULL);
    
    if(Monitor>=UTC_BASIC_INFO){
      printf("\n\tNumber of devices: \t%d\n", numDevices); 
    }
    // Iterate through each device, displaying associated information
    for (j = 0; j < numDevices; j++)
      {
	
	if(Monitor>UTC_BASIC_INFO){
	  printf("\tDevice %d:\n", j); 
	}
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].id = 
	  devices[j];
	clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(cl_device_type), 
	  &utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type, NULL);

	if(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type == CL_DEVICE_TYPE_CPU){
	  utv_ocl_struct.list_of_platforms[i].list_of_devices[j].utc_type = UTC_OCL_DEVICE_CPU;
	}
	if(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type == CL_DEVICE_TYPE_GPU){
	  utv_ocl_struct.list_of_platforms[i].list_of_devices[j].utc_type = UTC_OCL_DEVICE_GPU;
	}   
	if(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].type == CL_DEVICE_TYPE_ACCELERATOR){
	  utv_ocl_struct.list_of_platforms[i].list_of_devices[j].utc_type = UTC_OCL_DEVICE_ACCELERATOR;
	}  

	cl_ulong mem_size_ulong = 0;
	int err_num = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, 
			sizeof(cl_ulong), &mem_size_ulong, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_mem_bytes = 
	  (double)mem_size_ulong;

	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, 
			sizeof(cl_ulong), &mem_size_ulong, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_max_alloc= 
	  (double)mem_size_ulong;

	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_LOCAL_MEM_SIZE,
			sizeof(cl_ulong), &mem_size_ulong, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].local_mem_bytes = 
	  (double)mem_size_ulong;

	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
			sizeof(cl_ulong), &mem_size_ulong, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_bytes = 
	  (double)mem_size_ulong;

	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
			sizeof(cl_ulong), &mem_size_ulong, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].constant_mem_bytes = 
	  (double)mem_size_ulong;

	cl_uint cache_line_size = 0;
	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
			sizeof(cl_uint), &cache_line_size, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_line_bytes = 
	  (int) cache_line_size;

	cl_uint max_num_comp_units = 0;
	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS,
			sizeof(cl_uint), &max_num_comp_units, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_num_comp_units = 
	  (int) max_num_comp_units;

	size_t max_work_group_size =0;
	err_num = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE,
				  sizeof(size_t), &max_work_group_size, NULL);
	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_work_group_size = (int) max_work_group_size;

	// possible further inquires:
	//CL_DEVICE_MAX_WORK_GROUP_SIZE, 
	//CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, CL_DEVICE_MAX_WORK_ITEM_SIZES
	//CL_DEVICE_MAX_CONSTANT_ARGS
	//CL_DEVICE_MAX_PARAMETER_SIZE
	//CL_DEVICE_PREFERRED_VECTOR_WIDTH_ - char, int, float, double etc.
	//CL_DEVICE_MEM_BASE_ADDR_ALIGN, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE

	utv_ocl_struct.list_of_platforms[i].list_of_devices[j].command_queue = 0;
	
	for(k=0;k<UTC_OCL_MAX_NUM_KERNELS;k++){
	  utv_ocl_struct.list_of_platforms[i].list_of_devices[j].program[k]=0;
	  utv_ocl_struct.list_of_platforms[i].list_of_devices[j].kernel[k]=0;
	}
	      
	//clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(device_name?), 
	//&utv_ocl_struct.list_of_platforms[i].list_of_devices[j].name, NULL);

	if(Monitor>UTC_BASIC_INFO){

	  DisplayDeviceInfo(
			  devices[j], 
			  CL_DEVICE_NAME, 
			  "CL_DEVICE_NAME");
	
	  DisplayDeviceInfo(
			  devices[j], 
			  CL_DEVICE_VENDOR, 
			  "CL_DEVICE_VENDOR");
	
	  DisplayDeviceInfo(
			  devices[j], 
			  CL_DEVICE_VERSION, 
			  "CL_DEVICE_VERSION");
	  printf("\t\tdevice global memory size (MB) = %lf\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_mem_bytes/1024/1024);
	  printf("\t\tdevice global max alloc size (MB) = %lf\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].global_max_alloc/1024/1024);
	  printf("\t\tdevice local memory size (kB) = %lf\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].local_mem_bytes/1024);
	  printf("\t\tdevice constant memory size (kB) = %lf\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].constant_mem_bytes/1024);
	  printf("\t\tdevice cache memory size (kB) = %lf\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_bytes/1024);
	  printf("\t\tdevice cache line size (B) = %d\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].cache_line_bytes);
	  printf("\t\tdevice maximal number of compute units = %d\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_num_comp_units);
	  printf("\t\tdevice maximal number of work units in work group = %d\n",
		 utv_ocl_struct.list_of_platforms[i].list_of_devices[j].max_work_group_size);
	  
	  printf("\n");
	}
      }

    free(devices);
  
    // Next, create OpenCL contexts on platforms
    cl_context_properties contextProperties[] = {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)platformIds[i],
      0
    };

    if(Chosen_platform_id == UTC_OCL_ALL_PLATFORMS || Chosen_platform_id == i){

      if(Monitor>UTC_BASIC_INFO){
	printf("\tCreating CPU context (index=0) on platform %d\n", i);
      }

      utv_ocl_struct.list_of_platforms[i].list_of_contexts[0] = 
	clCreateContextFromType(contextProperties, 
				CL_DEVICE_TYPE_CPU, NULL, NULL, &retval);

      if(Monitor>=UTC_BASIC_INFO && retval != CL_SUCCESS){
	printf("\tCould not create CPU context on platform %d\n", i);
      }

      if(Monitor>UTC_BASIC_INFO){
	printf("\tCreating GPU context (index=1) on platform %d\n", i);
      }

      utv_ocl_struct.list_of_platforms[i].list_of_contexts[1] = 
	clCreateContextFromType(contextProperties, 
				CL_DEVICE_TYPE_GPU, NULL, NULL, &retval);

      if(Monitor>=UTC_BASIC_INFO && retval != CL_SUCCESS){
	printf("\tCould not create GPU context on platform %d\n", i);
      }

      if(Monitor>UTC_BASIC_INFO){
	printf("\tCreating ACCELERATOR context (index=2) on platform %d\n", i);
      }

      utv_ocl_struct.list_of_platforms[i].list_of_contexts[2] = 
	clCreateContextFromType(contextProperties, 
				CL_DEVICE_TYPE_ACCELERATOR, NULL, NULL, &retval);
      if(Monitor>=UTC_BASIC_INFO && retval != CL_SUCCESS){
	printf("\tCould not create ACCELERATOR context on platform %d\n", i);
      }

    }
  }
  
  free(platformIds);
  return numPlatforms;
}


int utr_ocl_create_command_queues(
    int Chosen_platform_index,
    int Chosen_device_type,
    int Monitor
  )
{

  // in a loop over all platforms
  int platform_index;
  for(platform_index=0; 
      platform_index<utv_ocl_struct.number_of_platforms; 
      platform_index++){

    // shortcut for global platform structure
    utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[platform_index];
    
    // if creating contexts for all platforms or just this one 
    if(Chosen_platform_index == UTC_OCL_ALL_PLATFORMS || 
       Chosen_platform_index == platform_index){
      
      // in a loop over all devices
      int idev;
      for(idev=0; 
	  idev<platform_struct.number_of_devices;
	  idev++){
	
	// variable for storing device_id
	cl_device_id device = 0;

	// select context for the device (CPU context for CPU device, etc.)
	// (contexts are already created!,
	// icon is just the index in the platform structure)	
	int icon;
	
	// check whether this is a CPU device - then context is no 0
	if(platform_struct.list_of_devices[idev].type ==
	   CL_DEVICE_TYPE_CPU){
	  
	  if(Chosen_device_type == UTC_OCL_ALL_DEVICES || 
	     Chosen_device_type == UTC_OCL_DEVICE_CPU){
	    
	    device = platform_struct.list_of_devices[idev].id;
	    platform_struct.list_of_devices[idev].utc_type = UTC_OCL_DEVICE_CPU;
	    icon = 0;
	    
	  }
	  else{
	    
	    device = NULL;
	    
	  }
	  
	}
	// check whether this is a GPU device - then context is no 1
	else if(platform_struct.list_of_devices[idev].type ==
		CL_DEVICE_TYPE_GPU){
	  
	  if(Chosen_device_type == UTC_OCL_ALL_DEVICES || 
	     Chosen_device_type == UTC_OCL_DEVICE_GPU){
	    
	    device = platform_struct.list_of_devices[idev].id;
	    platform_struct.list_of_devices[idev].utc_type = UTC_OCL_DEVICE_GPU;
	    icon = 1;
	    
	  }
	  else{
	    
	    device = NULL;
	    
	  }
	  
	}
	// check whether this is an ACCELERATOR device - then context is no 2
	else if(platform_struct.list_of_devices[idev].type ==
		CL_DEVICE_TYPE_ACCELERATOR){
	  
	  if(Chosen_device_type == UTC_OCL_ALL_DEVICES || 
	     Chosen_device_type == UTC_OCL_DEVICE_ACCELERATOR){
	    
	    device = platform_struct.list_of_devices[idev].id;
	    platform_struct.list_of_devices[idev].utc_type = UTC_OCL_DEVICE_ACCELERATOR;
	    icon = 2;
	    
	  }
	  else{
	    
	    device = NULL;
	    
	  }
	  
	}
	
	if(device != NULL){
	  
	  // choose OpenCL context selected for a device
	  cl_context context = platform_struct.list_of_contexts[icon];
	  platform_struct.list_of_devices[idev].context_index = icon;
	  
	  // if context exist
	  if(context != NULL){
	    
	    if(Monitor>UTC_BASIC_INFO){
	      if(platform_struct.list_of_devices[idev].utc_type == UTC_OCL_DEVICE_CPU){
		printf("\nCreating command queue for CPU context %d, device %d, platform %d\n",
		       icon, idev, platform_index);
	      }
	      if(platform_struct.list_of_devices[idev].utc_type == UTC_OCL_DEVICE_GPU){
		printf("\nCreating command queue for GPU context %d, device %d, platform %d\n",
		       icon, idev, platform_index);
	      }
	      if(platform_struct.list_of_devices[idev].utc_type == UTC_OCL_DEVICE_ACCELERATOR){
		printf("\nCreating command queue for ACCELERATOR context %d, device %d, platform %d\n",
		       icon, idev, platform_index);
	      }
	    }

	    // Create a command-queue on the device for the context
	    cl_command_queue_properties prop = 0;
	    prop |= CL_QUEUE_PROFILING_ENABLE;
	    platform_struct.list_of_devices[idev].command_queue = 
	      clCreateCommandQueue(context, device, prop, NULL);
	    if (platform_struct.list_of_devices[idev].command_queue == NULL)
	      {
		printf("Failed to create command queue for context %d, device %d, platform %d\n",
		       icon, idev, platform_index);
		exit(-1);
	      }
	    
	  } // end if context exist for a given device
	  
	} // end if device is of specified type
	
      } // end loop over devices
      
    } // end if platform is of specified type
    
  } // end loop over platforms
  
  return(0);
}

// auxiliary procedure for reading source files
char* utr_ocl_readSource(const char* kernelPath) {

   cl_int status;
   FILE *fp;
   char *source;
   long int size;

   fp = fopen(kernelPath, "rb");
   if(!fp) {
      printf("Could not open kernel file\n");
      exit(-1);
   }
   status = fseek(fp, 0, SEEK_END);
   if(status != 0) {
      printf("Error seeking to end of file\n");
      exit(-1);
   }
   size = ftell(fp);
   if(size < 0) {
      printf("Error getting file position\n");
      exit(-1);
   }

   rewind(fp);

   source = (char *)malloc(size + 1);

   int i;
   for (i = 0; i < size+1; i++) {
      source[i]='\0';
   }

   if(source == NULL) {
      printf("Error allocating space for the kernel source\n");
      exit(-1);
   }

   fread(source, 1, size, fp);
   source[size] = '\0';

   return source;
}

int utr_ocl_create_kernel(
  int Platform_index,
  int Device_index,
  int Kernel_index,
  char* Kernel_name,
  const char* Kernel_file,
  int Monitor
)
{

  cl_int retval;

  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];

  // check the device !!!!!!!!!!!!!!!!! (or at least its index)
  if(Device_index < 0){
    printf("Wrong device_index %d passed to utr_ocl_create_kernel! Exiting.\n",
	   Device_index);
    exit(-1);
  } 

  cl_device_id device = platform_struct.list_of_devices[Device_index].id;
  cl_context context = platform_struct.list_of_contexts[ 
			 platform_struct.list_of_devices[Device_index].context_index
							 ];

  if(Monitor>UTC_BASIC_INFO){
    printf("Program file is: %s\n", Kernel_file);
  }

  // read source file into data structure
  const char* source = utr_ocl_readSource(Kernel_file);



  cl_program program = clCreateProgramWithSource(context, 1,
				      &source,
				      NULL, NULL);
  if (program == NULL)
    {
      printf("Failed to create CL program from source.\n");
      exit(-1);
    }
  
  // TO GET INFO FROM NVIDIA COMPILER
  //retval = clBuildProgram(program, 0, NULL, "-cl-nv-verbose", NULL, NULL); 
  retval = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  char* buildLog; size_t size_of_buildLog; 
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
			0, NULL, &size_of_buildLog); 
  buildLog = malloc(size_of_buildLog+1); 
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 
			size_of_buildLog, buildLog, NULL); 
  buildLog[size_of_buildLog]= '\0'; 
  printf("Kernel buildLog: %s\n", buildLog); 
  if (retval != CL_SUCCESS)
    {
      // Determine the reason for the error
      char buildLog[16384];
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
			    sizeof(buildLog), buildLog, NULL);
      
      printf("Error in kernel\n");
      clReleaseProgram(program);
      exit(-1);
      //return NULL;
    }


  // Create OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, Kernel_name, NULL);
  if (kernel == NULL)
    {
      printf("Failed to create kernel.\n");
      exit(-1);
      //  return 1;
    }
  
  if(Monitor>UTC_BASIC_INFO){
    printf("Created kernel for platform %d, device %d, kernel index %d\n",
	   Platform_index, Device_index, Kernel_index);
  }
  
  platform_struct.list_of_devices[Device_index].program[Kernel_index] = program;
  platform_struct.list_of_devices[Device_index].kernel[Kernel_index] = kernel;
  
  return(0);
}

int utr_ocl_create_kernel_dev_type(
  int Platform_index,
  int Device_type,
  int Kernel_index,
  char* Kernel_name,
  const char* Kernel_file,
  int Monitor
)
{

  cl_int retval;

  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];

  // choose the device
  int idev; int device_index;
  for(idev=0; 
      idev<platform_struct.number_of_devices;
      idev++){

    if(platform_struct.list_of_devices[idev].utc_type == Device_type){

      if(Monitor>UTC_BASIC_INFO){
	if(Device_type==UTC_OCL_DEVICE_CPU){
	  printf("\nCreating kernel %d for platform %d: selected device %d for type %d (CPU)\n",
		 Kernel_index, Platform_index, idev, Device_type);
	}
	if(Device_type==UTC_OCL_DEVICE_GPU){
	  printf("\nCreating kernel %d for platform %d: selected device %d for type %d (GPU)\n",
	       Kernel_index, Platform_index, idev, Device_type);
	}
	if(Device_type==UTC_OCL_DEVICE_ACCELERATOR){
	  printf("\nCreating kernel %d for platform %d: selected device %d for type %d (ACCELERATOR)\n",
	       Kernel_index, Platform_index, idev, Device_type);
	}
      }

      device_index = idev;
      break;

    }

  }

  utr_ocl_create_kernel(Platform_index, device_index, Kernel_index, 
			Kernel_name, Kernel_file, Monitor);

  return(0);
}

// returns OpenCL device index (for local data structures) or -1 if device
// is not available (not existing or not serviced) for the specified platform
int utr_ocl_select_device( 
			  int Platform_index,
			  int Device_utc_type
			   )
{
  int device_index = -1;
  
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  
  // in a loop over all devices
  int idev;
  for(idev=0; idev<platform_struct.number_of_devices; idev++){

    //printf("platform %d, idev %d, type %d, input_type %d\n",
    //	   Platform_index, idev, platform_struct.list_of_devices[idev].utc_type, Device_utc_type);
	
    // check device type
    if(platform_struct.list_of_devices[idev].utc_type == Device_utc_type) {
      device_index = idev;
      break;
    }
  }

  return(device_index);
}


int utr_ocl_device_type(
  int Platform_index,
  int Device_index
  )
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  return(platform_struct.list_of_devices[Device_index].utc_type);
}

cl_context utr_ocl_select_context(
  int Platform_index,
  int Device_index
)
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  int context_index = platform_struct.list_of_devices[Device_index].context_index;
  return(platform_struct.list_of_contexts[context_index]);
}

int utr_ocl_CPU_context_exists(
  int Platform_index
)
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  if(platform_struct.list_of_contexts[0]==NULL) return(0);
  return(1);
}

int utr_ocl_GPU_context_exists(
  int Platform_index
)
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  if(platform_struct.list_of_contexts[1]==NULL) return(0);
  return(1);
}

int utr_ocl_ACCELERATOR_context_exists(
  int Platform_index
)
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  if(platform_struct.list_of_contexts[2]==NULL) return(0);
  return(1);
}

cl_command_queue utr_ocl_select_command_queue(
  int Platform_index,
  int Device_index
)
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  return(platform_struct.list_of_devices[Device_index].command_queue);
}

cl_kernel utr_ocl_select_kernel(
  int Platform_index,
  int Device_index,
  int Kernel_index
)
{
  // choose the platform
  utt_ocl_platform_struct platform_struct = utv_ocl_struct.list_of_platforms[Platform_index];
  return(platform_struct.list_of_devices[Device_index].kernel[Kernel_index]);
}

double utr_ocl_calculate_execution_time(
  cl_event ndrEvt
				  )
{
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
  double time = ((double)endTime - (double)startTime)*1.0e-9;

  return(time);
}

///
//  Cleanup any created OpenCL resources
//
void utr_ocl_cleanup()
{
  int i,j,k;

  for(i=0; i< utv_ocl_struct.number_of_platforms; i++){
    
    for(j=0; j<utv_ocl_struct.list_of_platforms[i].number_of_devices; j++){
      
      if(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].command_queue!=0){
        clReleaseCommandQueue(
		      utv_ocl_struct.list_of_platforms[i].list_of_devices[j].command_queue);
      }
      
      for(k=0;k<UTC_OCL_MAX_NUM_KERNELS;k++){
	if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].kernel[k] != 0){
	  clReleaseKernel(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].kernel[k]);
	}
      }
      
      for(k=0;k<UTC_OCL_MAX_NUM_KERNELS;k++){
	if (utv_ocl_struct.list_of_platforms[i].list_of_devices[j].program[k] != 0){
	  clReleaseProgram(utv_ocl_struct.list_of_platforms[i].list_of_devices[j].program[k]);
	}
      }
      
    }
     
    free(utv_ocl_struct.list_of_platforms[i].list_of_devices);
    
    if (utv_ocl_struct.list_of_platforms[i].list_of_contexts[0] != 0)
      clReleaseContext(utv_ocl_struct.list_of_platforms[i].list_of_contexts[0]);
    
    if (utv_ocl_struct.list_of_platforms[i].list_of_contexts[1] != 0)
      clReleaseContext(utv_ocl_struct.list_of_platforms[i].list_of_contexts[1]);
    
    if (utv_ocl_struct.list_of_platforms[i].list_of_contexts[2] != 0)
      clReleaseContext(utv_ocl_struct.list_of_platforms[i].list_of_contexts[2]);
    
  }
  
  free(utv_ocl_struct.list_of_platforms);
    
}
  
  
