#SHELL = /bin/csh
SHELL = /bin/bash

all: config OpenCL_info

config:
	@echo 'Configuring obj and bin directories... '
	cd ..; $(MAKE)


SRR_OBJ_DIR = $(SRR_DIR)/obj/$(SRR_ARCH)
SRR_BIN_DIR = $(SRR_DIR)/bin/$(SRR_ARCH)
SRR_INC_DIR = $(SRR_DIR)/src/include
SRR_LIB_DIR = $(SRR_DIR)/src/lib


# libraries with compiled modules:
SRR_system_util_LIB = $(SRR_OBJ_DIR)/system_util_lib.a
SRR_OpenCL_util_LIB = $(SRR_OBJ_DIR)/OpenCL_util_lib.a

# optimization and other platform dependent options and definitions
include  ../platform_files/make.$(SRR_ARCH)

export

# programs:
OpenCL_info: $(SRR_OBJ_DIR)/OpenCL_info.o
	cd ../system_util; $(MAKE)
	cd ../OpenCL_util; $(MAKE)
	$(LD) $(LDFL) \
	-o $(SRR_DIR)/bin/$(SRR_ARCH)/SRR_ocl_OpenCL_info.bin \
	$(SRR_OBJ_DIR)/OpenCL_info.o \
	$(SRR_system_util_LIB) \
	$(SRR_OpenCL_util_LIB) \
	$(LIB)
	@echo 'SRR executable for "OpenCL_info":'
	@echo '$(SRR_BIN_DIR)/SRR_ocl_OpenCL_info.bin'


#create necessary obj files
$(SRR_OBJ_DIR)/OpenCL_info.o : OpenCL_info.c
	$(CC) $(CFL) -c OpenCL_info.c -o $@ $(INC)

clean:
	rm -f $(SRR_OBJ_DIR)/*.o  

deep_clean: 
	rm -f $(SRR_OBJ_DIR)/*.o  
	rm -f $(SRR_OBJ_DIR)/*.a  
