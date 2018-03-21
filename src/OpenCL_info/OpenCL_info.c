#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "omp.h"
#include "../include/OpenCL_util.h"

#include <CL/cl.h>

void displayInfo();

int main(int argc, char *argv[])
{

  printf("\nOpenCL - DISPLAY INFORMATION ABOUT SOFTWARE and HARDWARE\n");
  displayInfo();
  return(0);
}

void displayInfo()
{
  int i,j;
  cl_int retval;
  char *info; size_t size;

  /* POBRANIE I WYSWIETLENIE LICZBY PLATFORM */
  // First, query the total number of platforms
  cl_uint numPlatforms;
  retval = clGetPlatformIDs(0, (cl_platform_id *) NULL, &numPlatforms);
  printf("\nNumber of platforms: %u\n", numPlatforms);

  /* POBRANIE I WYSWIETLENIE INFORMACJI O PLATFORMACH */
  // Next, allocate memory for the installed plaforms, and qeury
  // to get the list.
  cl_platform_id * platformIds;
  platformIds = (cl_platform_id *)malloc(sizeof(cl_platform_id) * numPlatforms);

  // Then, query the platform IDs
  retval = clGetPlatformIDs(numPlatforms, platformIds, NULL);

  // Iterate through the list of platforms displaying associated information
  for (i = 0; i < numPlatforms; i++) {

      printf("\nPlatform ID - %d\n",i);

    //Nazwa
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, 0, NULL, &size);
    info = (char*) malloc (size*sizeof(char));
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_NAME, size, info, NULL);
    printf("\nPlatform name: ---------------------------------- %s",info);
    free(info);

    //Nazwa producenta
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, 0, NULL, &size);
    info = (char*) malloc (size*sizeof(char));
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VENDOR, size, info, NULL);
    printf("\nVendor name: ------------------------------------ %s",info);
    free(info);

    //Wersja platformy
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, 0, NULL, &size);
    info = (char*) malloc (size*sizeof(char));
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_VERSION, size, info, NULL);
    printf("\nVersion: ---------------------------------------- %s",info);
    free(info);

    //Informacje o profilu wsparcie opencl
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, 0, NULL, &size);
    info = (char*) malloc (size*sizeof(char));
    retval = clGetPlatformInfo(platformIds[i], CL_PLATFORM_PROFILE, size, info, NULL);
    printf("\nPlatform profile: ------------------------------- %s:",info);
    free(info);

    /* POBRANIE I WYSWIETLENIE LISTY URZADZEN */
    cl_uint numDevices;
    retval = clGetDeviceIDs(platformIds[i],CL_DEVICE_TYPE_ALL,0,NULL,&numDevices);
    printf("\n\tNumber of devices: %u\n",numDevices);

    /* POBRANIE INFORMACJI O URZADZENIACH */
    cl_device_id *devicesIds;
    devicesIds = (cl_device_id*) malloc (numDevices*sizeof(cl_device_id));
    retval = clGetDeviceIDs(platformIds[i],CL_DEVICE_TYPE_ALL,numDevices,devicesIds,NULL);
    for(j=0; j<numDevices; j++) {

      //Nazwa
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_NAME,0,NULL,&size);
      info = (char*) malloc (size*sizeof(char));
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_NAME,size,info,&size);
      printf("\nDevice name: ------------------------------------ %s",info);
      free(info);

      //Dostawca
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_VENDOR,0,NULL,&size);
      info = (char*) malloc (size*sizeof(char));
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_VENDOR,size,info,&size);
      printf("\nDevice vendor: ---------------------------------- %s",info);
      free(info);

      //Dostepny sterownik OpenCL
      retval = clGetDeviceInfo(devicesIds[j],CL_DRIVER_VERSION,0,NULL,&size);
      info = (char*) malloc (size*sizeof(char));
      retval = clGetDeviceInfo(devicesIds[j],CL_DRIVER_VERSION,size,info,&size);
      printf("\nDriver version: --------------------------------- %s",info);
      free(info);

      //Wspierany OpenCL
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_VERSION,0,NULL,&size);
      info = (char*) malloc (size*sizeof(char));
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_VERSION,size,info,&size);
      printf("\nSupported OpenCL by device: --------------------- %s",info);
      free(info);

      //Type
      cl_device_type infoType;
      retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_TYPE,sizeof(cl_device_type),&infoType,NULL);
      switch(infoType)
	{
	case CL_DEVICE_TYPE_CPU:
	  printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_CPU");
	  break;

	case CL_DEVICE_TYPE_GPU:
	  printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_GPU");
	  break;

	case CL_DEVICE_TYPE_ACCELERATOR:
	  printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_ACCELERATOR");
	  break;

	default:
	  printf("\nDevice type: ------------------------------------ CL_DEVICE_TYPE_DEFAULT");
	  break;
	};


      //Profil urzadzenia
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PROFILE,0,NULL,&size);
		info = (char*) malloc (size*sizeof(char));
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PROFILE,size,info,&size);
		printf("\nProfil urządzenia: --------------------- %s",info);
		free(info);

		cl_platform_id infoPlatformId;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PRINTF_BUFFER_SIZE,sizeof(size_t),&infoPlatformId,NULL);
		printf("\nDevice platform id: --------------------- %p",infoPlatformId);

		size_t infoPrintfBufferSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PRINTF_BUFFER_SIZE,sizeof(size_t),&infoPrintfBufferSize,NULL);
		printf("\nDevice printf buffer size: --------------------- %zu",infoPrintfBufferSize);

		cl_bool infoPrefIntUsrSync;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_INTEROP_USER_SYNC,sizeof(cl_bool),&infoPrefIntUsrSync,NULL);
		printf("\nDevice prefered interop user sync: --------------------- %u",infoPrefIntUsrSync);

		//Rozmiar pamieci globalnej-----------------
		cl_ulong infoGlobMemSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_GLOBAL_MEM_SIZE,sizeof(cl_ulong),&infoGlobMemSize,NULL);
		printf("\nDevice gloabl memory size: ---------------------- %lf [MB]",(((double) infoGlobMemSize)/1048576.0));


		//Rozmiar cache pamieci globalnej-----------------
		cl_ulong infoGlobCacheSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(cl_ulong),&infoGlobCacheSize,NULL);
		printf("\nDevice global memory cache size: ---------------- %lf [KB]",(((double) infoGlobCacheSize)/1024.0));

		//Rozmiar linijki cache pamieci globalnej---------------
		cl_uint infoGlobCacheLineSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,sizeof(cl_uint),&infoGlobCacheLineSize,NULL);
		printf("\nDevice global memory cache line size: ----------- %u [B]",infoGlobCacheLineSize);

		//Rozmiar pamieci lokalnej----------------
		cl_ulong infoLocalMemSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_LOCAL_MEM_SIZE,sizeof(cl_ulong),&infoLocalMemSize,NULL);
		printf("\nDevice local memory size: ----------------------- %lf [KB]",(((double) infoLocalMemSize)/1024.0));

		//Maksymalny rozmiar pamięci do zaalokowania-----------
		cl_ulong infoMaxMemAllocSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&infoMaxMemAllocSize,NULL);
		printf("\nDevice max memory allocation size: -------------- %lf [MB]",(((double) infoMaxMemAllocSize)/1048576.0));

		//Maksymalny rozmiar bufora stałych--------------
		cl_ulong infoMaxConstBufSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,sizeof(cl_ulong),&infoMaxConstBufSize,NULL);
		printf("\nDevice max constant buffer size: ---------------- %lf [KB]",(((double) infoMaxConstBufSize)/1024.0));


		//Maksymalna liczba jednostek obliczeniowych-------------
		cl_uint infoMaxCompUnits;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cl_uint),&infoMaxCompUnits,NULL);
		printf("\nDevice max compute units: ----------------------- %u",infoMaxCompUnits);

		cl_uint infoMaxConstantArgs;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MAX_CONSTANT_ARGS,sizeof(cl_uint),&infoMaxConstantArgs,NULL);
		printf("\nDevice max constant args: --------------------- %u",infoMaxConstantArgs);

		size_t infoMaxParameterSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MAX_PARAMETER_SIZE,sizeof(size_t),&infoMaxParameterSize,NULL);
		printf("\nDevice max parameter size: --------------------- %zu",infoMaxParameterSize);

		cl_uint infoMemBaseAddrAlign;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MEM_BASE_ADDR_ALIGN,sizeof(cl_uint),&infoMemBaseAddrAlign,NULL);
		printf("\nDevice mem base addr align: --------------------- %u",infoMemBaseAddrAlign);

		cl_uint infoMinDataTypeAlignSize;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,sizeof(cl_uint),&infoMinDataTypeAlignSize,NULL);
		printf("\nDevice min data type align size: --------------------- %u",infoMinDataTypeAlignSize);

		cl_uint infoPreferredVectorWidth;

		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width short: --------------------- %u",infoPreferredVectorWidth);
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width char: --------------------- %u",infoPreferredVectorWidth);
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width int: --------------------- %u",infoPreferredVectorWidth);
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width long: --------------------- %u",infoPreferredVectorWidth);
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width float: --------------------- %u",infoPreferredVectorWidth);
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width double: --------------------- %u",infoPreferredVectorWidth);
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF,sizeof(cl_uint),&infoPreferredVectorWidth,NULL);
		printf("\nDevice preferred vector width half: --------------------- %u",infoPreferredVectorWidth);

		cl_device_partition_property *partitionInfo;
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PARTITION_TYPE,0,NULL,&size);
		partitionInfo = (cl_device_partition_property *) malloc (size*sizeof(cl_device_partition_property));
		retval = clGetDeviceInfo(devicesIds[j],CL_DEVICE_PARTITION_TYPE,size,partitionInfo,&size);
		printf("\nTypy partycji: ---------------------");

		if(size == 0) {
			printf("There is no partition type associated with device");
		}

		for (int k = 0; k < size; ++k) {



			switch (partitionInfo[k]) {
				case CL_DEVICE_PARTITION_EQUALLY:
					printf("\nPartition type: ----------------------- CL_DEVICE_PARTITION_EQUALLY");
					break;

				case CL_DEVICE_PARTITION_BY_COUNTS:
					printf("\nDevice local memory type: ----------------------- CL_GLOBAL");
					break;

				case CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN:
					printf("\nDevice local memory type: ----------------------- CL_DEVICE_PARTITION_BY_AFFINITY_DOMAIN");
					break;

				default:
					printf("\nDevice local memory type: ----------------------- %p", partitionInfo);
					break;
			};

		}
		free(partitionInfo);



    }
    free(devicesIds);

    printf("\n\n\t ---- ************* ----\n");
  }
  printf("\n\n");
  free(platformIds);

  // create contexts for all platforms
  int platform_id = -1; // according to convention -1 means all platforms
  int monitor = 3;  // according to convention - print all info
  utr_ocl_create_contexts(platform_id, monitor);

}
