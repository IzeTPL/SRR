/******************************************************************************
File uth_time_intf.h - interface to utility procedures (and macros)

Time measurments:
  time_C - (C standard procedure) to return time in seconds from some date

  time_init   - to initiate time measurments
  time_clock  - to return wall clock time from initialization
  time_CPU    - to return CPU  time from initialization
  time_print  - to print CPU and wall clock time from initialization

******************************************************************************/

#ifndef _uth_time_intf_
#define _uth_time_intf_

/* standard macro for max and min and abs */
#define utm_max(x,y) ((x)>(y)?(x):(y))
#define utm_min(x,y) ((x)<(y)?(x):(y))
#define utm_abs(x)   ((x)<0?-(x):(x))

#ifdef __cplusplus
extern "C"
{
#endif


/*---------------------------------------------------------
  time_init   - to initiate time measurments
---------------------------------------------------------*/
void time_init();

/*---------------------------------------------------------
  time_C - (C standard procedure) to return time in seconds from some date
---------------------------------------------------------*/
double time_C();

/*---------------------------------------------------------
  time_clock  - to return wall clock time from initialization
---------------------------------------------------------*/
double time_clock();

/*---------------------------------------------------------
  time_CPU    - to return CPU  time from initialization
 ---------------------------------------------------------*/
double time_CPU();

/*---------------------------------------------------------
  time_print  - to print CPU and wall clock time from initialization
 ---------------------------------------------------------*/
void time_print();

#ifdef __cplusplus
}
#endif

#endif
