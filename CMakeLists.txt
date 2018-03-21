cmake_minimum_required(VERSION 3.9)
project(OpenCL_codes)

set(CMAKE_CXX_STANDARD 11)

include_directories(/include)

add_executable(OpenCL_codes
        src/bin/intel/SRR_ocl_OpenCL_info.bin
        src/include/OpenCL_util.h
        src/include/system_util.h
        src/obj/intel/OpenCL_info.o
        src/obj/intel/system_util.o
        src/obj/intel/system_util_lib.a
        src/OpenCL_info/OpenCL_info.c
        src/OpenCL_util/OpenCL_util.c
        src/platform_files/make.centos7_nv_ocl
        src/platform_files/make.intel
        src/platform_files/make.l404_amd
        src/platform_files/make.l404_nvidia
        src/system_util/system_util.c
        src/main_driver.c
        src/Makefile
        Makefile)
