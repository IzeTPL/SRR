$(SRR_DIR)/obj/$(SRR_ARCH)/OpenCL_util_lib.a : \
           $(SRR_DIR)/obj/$(SRR_ARCH)/OpenCL_util.o
	$(AR) $@ $(SRR_DIR)/obj/$(SRR_ARCH)/OpenCL_util.o

$(SRR_DIR)/obj/$(SRR_ARCH)/OpenCL_util.o: OpenCL_util.c
	$(CC) $(CFL) -c OpenCL_util.c -o $@ $(INC)

