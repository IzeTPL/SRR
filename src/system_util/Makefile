$(SRR_DIR)/obj/$(SRR_ARCH)/system_util_lib.a : \
           $(SRR_DIR)/obj/$(SRR_ARCH)/system_util.o
	$(AR) $@ $(SRR_DIR)/obj/$(SRR_ARCH)/system_util.o

$(SRR_DIR)/obj/$(SRR_ARCH)/system_util.o: system_util.c
	$(CC) $(CFL) -c system_util.c -o $@ $(INC)

