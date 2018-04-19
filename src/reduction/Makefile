SHELL = /bin/csh
#SHELL = /bin/sh

all: config reduction

config:
	@echo 'Configuring obj and bin directories... '
	cd ..; $(MAKE)


SRR_OBJ_DIR = $(SRR_DIR)/obj/$(SRR_ARCH)
SRR_BIN_DIR = $(SRR_DIR)/bin/$(SRR_ARCH)
SRR_INC_DIR = $(SRR_DIR)/src/include
SRR_LIB_DIR = $(SRR_DIR)/src/lib


# libraries with compiled modules:
SRR_OpenCL_util_LIB = $(SRR_OBJ_DIR)/OpenCL_util_lib.a
SRR_system_util_LIB = $(SRR_OBJ_DIR)/system_util_lib.a

# optimization and other platform dependent options and definitions
include  ../platform_files/make.$(SRR_ARCH)

export

# programs:
reduction: $(SRR_OBJ_DIR)/main_driver.o $(SRR_OBJ_DIR)/reduction_host.o \
	      $(SRR_OBJ_DIR)/work_reduction.o
	cd ../OpenCL_util; $(MAKE)
	cd ../system_util; $(MAKE)
	$(LD) $(LDFL) \
	-o $(SRR_DIR)/bin/$(SRR_ARCH)/SRR_ocl_reduction.bin \
	$(SRR_OBJ_DIR)/main_driver.o \
	$(SRR_OBJ_DIR)/reduction_host.o \
	$(SRR_OBJ_DIR)/work_reduction.o \
	$(SRR_OpenCL_util_LIB) \
	$(SRR_system_util_LIB) \
	$(LIB)
	@echo 'SRR executable for "reduction":'
	@echo '$(SRR_BIN_DIR)/SRR_ocl_reduction.bin'


#create necessary obj files
$(SRR_OBJ_DIR)/main_driver.o : ../main_driver.c
	$(CC) $(CFL) -c ../main_driver.c -o $@ $(INC)

$(SRR_OBJ_DIR)/reduction_host.o : reduction_host.c
	$(CC) $(CFL) -c reduction_host.c -o $@ $(INC)

$(SRR_OBJ_DIR)/work_reduction.o : work_reduction.c
	$(CC) $(CFL) -c work_reduction.c -o $@ $(INC)

clean:
	rm -f $(SRR_OBJ_DIR)/*.o  

deep_clean: 
	rm -f $(SRR_OBJ_DIR)/*.o  
	rm -f $(SRR_OBJ_DIR)/*.a  
