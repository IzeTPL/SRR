// for single or double precision calculations
// do not forget to choose the proper kernel!
//#define SCALAR double
#define SCALAR float

extern int verify_result(
		  int size,
		  SCALAR* result,
		  SCALAR* result_compare
		  );

extern int vec_add_host(
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
		      );

extern int vec_add_host_2(
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
			); 
