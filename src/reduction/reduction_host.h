// for single or double precision calculations
// do not forget to choose the proper kernel!
//#define SCALAR double
#define SCALAR double

extern int verify_result(
		  SCALAR result,
		  SCALAR result_compare
		  );

extern int reduction_host(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* Sc_prod, 
  int N, 
  int Nr_threads, 
  int Work_group_size, 
  const cl_context context, 
  const cl_kernel reduction_kernel, 
  const cl_command_queue queue
		      );

extern int reduction_host_2(
  int kernel_index,
  SCALAR* A, 
  SCALAR* B, 
  SCALAR* Sc_prod, 
  int N, 
  int Nr_threads, 
  int Work_group_size, 
  const cl_context context, 
  const cl_kernel reduction_kernel, 
  const cl_command_queue queue
			); 
