SHELL = /bin/csh
#SHELL = /bin/sh

all: config mat_transp

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
mat_transp: $(SRR_OBJ_DIR)/main_driver.o $(SRR_OBJ_DIR)/mat_transp_host.o
	cd ../OpenCL_util; $(MAKE)
	cd ../system_util; $(MAKE)
	$(LD) $(LDFL) \
	-o $(SRR_DIR)/bin/$(SRR_ARCH)/SRR_ocl_mat_transp.bin \
	$(SRR_OBJ_DIR)/main_driver.o $(SRR_OBJ_DIR)/mat_transp_host.o \
	$(SRR_OpenCL_util_LIB) \
	$(SRR_system_util_LIB) \
	$(LIB)
	@echo 'SRR executable for "mat_transp":'
	@echo '$(SRR_BIN_DIR)/SRR_ocl_mat_transp.bin'


#create necessary obj files
$(SRR_OBJ_DIR)/main_driver.o : ../main_driver.c
	$(CC) $(CFL) -c ../main_driver.c -o $@ $(INC)

$(SRR_OBJ_DIR)/mat_transp_host.o : mat_transp_host.c
	$(CC) $(CFL) -c mat_transp_host.c -o $@ $(INC)

clean:
	rm -f $(SRR_OBJ_DIR)/*.o  

deep_clean: 
	rm -f $(SRR_OBJ_DIR)/*.o  
	rm -f $(SRR_OBJ_DIR)/*.a  
