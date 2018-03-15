#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "omp.h"

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

      //Maksymalna czestotliwosc zegara MHz

      //Rozmiar pamieci globalnej

      //Rozmiar cache pamieci globalnej

      //Rozmiar linijki cache pamieci globalnej

      //Typ pamieci lokalnej

      //Rozmiar pamieci lokalnej

      //Maksymalny rozmiar pamięci do zaalokowania

      //Maksymalny rozmiar bufora stałych

      //Maksymalna liczba jednostek obliczeniowych

      //Maksymalny rozmiar grupy roboczej

      //Maksymalny wymiar przestrzeni wątków

      //Rozszerzenia


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
