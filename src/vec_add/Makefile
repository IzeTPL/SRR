SHELL = /bin/csh
#SHELL = /bin/sh

all: config vec_add

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
vec_add: $(SRR_OBJ_DIR)/main_driver.o $(SRR_OBJ_DIR)/vec_add_host.o \
	      $(SRR_OBJ_DIR)/work_vec_add.o
	cd ../OpenCL_util; $(MAKE)
	cd ../system_util; $(MAKE)
	$(LD) $(LDFL) \
	-o $(SRR_DIR)/bin/$(SRR_ARCH)/SRR_ocl_vec_add.bin \
	$(SRR_OBJ_DIR)/main_driver.o \
	$(SRR_OBJ_DIR)/vec_add_host.o \
	$(SRR_OBJ_DIR)/work_vec_add.o \
	$(SRR_OpenCL_util_LIB) \
	$(SRR_system_util_LIB) \
	$(LIB)
	@echo 'SRR executable for "vec_add":'
	@echo '$(SRR_BIN_DIR)/SRR_ocl_vec_add.bin'


#create necessary obj files
$(SRR_OBJ_DIR)/main_driver.o : ../main_driver.c
	$(CC) $(CFL) -c ../main_driver.c -o $@ $(INC)

$(SRR_OBJ_DIR)/vec_add_host.o : vec_add_host.c
	$(CC) $(CFL) -c vec_add_host.c -o $@ $(INC)

$(SRR_OBJ_DIR)/work_vec_add.o : work_vec_add.c
	$(CC) $(CFL) -c work_vec_add.c -o $@ $(INC)

clean:
	rm -f $(SRR_OBJ_DIR)/*.o  

deep_clean: 
	rm -f $(SRR_OBJ_DIR)/*.o  
	rm -f $(SRR_OBJ_DIR)/*.a  
