#SHELL = /bin/csh
SHELL = /bin/bash


# required environment variables 
# (specified in shell start files, by shell commands or here in Makefile):

# SRR_DIR - directory for SRR lab codes; 
#override SRR_DIR ?= /home/marian/ClionProjects/OpenCL_codes/src

# SRR_ARCH - architecture (platform) name for local definitions;
#            During compilation file make.SRR_ARCH placed in SRR_DIR/src/platform_files  
#            is read to set system dependent options. 
#override SRR_ARCH ?= l404_amd

# obj and binary files are in separate directories for each platform.
# SRR_DIR/obj/SRR_ARCH directory for obj files
# SRR_DIR/bin/SRR_ARCH directory for binary files

all: config 

config:
ifeq ($(origin SRR_DIR), undefined)
$(error No SRR_DIR system variable specified for SRR code directory.\
 Set SRR_DIR system variable to the name of the directory\
 where the current src directory resides - see Makefile for why and how to do it)
endif
ifeq ($(origin SRR_ARCH), undefined)
$(error No SRR_ARCH system variable specified for system dependent compilation. \
 Set SRR_ARCH system variable to some arbitrarily chosen platform name\
 corresponding to the file with local definitions in src/platform_files directory \
 - see Makefile for why and how to do it) 
endif

ifeq ($(SHELL),/bin/csh)
	@if (!( -d $(SRR_DIR)/bin )) mkdir $(SRR_DIR)/bin
	@if (!( -d $(SRR_DIR)/obj )) mkdir $(SRR_DIR)/obj
	@if (!( -d $(SRR_DIR)/bin/$(SRR_ARCH) )) mkdir $(SRR_DIR)/bin/$(SRR_ARCH)
	@if (!( -d $(SRR_DIR)/obj/$(SRR_ARCH) )) mkdir $(SRR_DIR)/obj/$(SRR_ARCH)
endif

ifeq ($(SHELL),/bin/bash)
	@if [[ ! -d "$(SRR_DIR)/bin" ]] ; then mkdir "$(SRR_DIR)/bin" ; fi
	@if [[ ! -d "$(SRR_DIR)/obj" ]] ; then mkdir "$(SRR_DIR)/obj" ; fi
	@if [[ ! -d "$(SRR_DIR)/bin/$(SRR_ARCH)" ]] ; then mkdir "$(SRR_DIR)/bin/$(SRR_ARCH)" ; fi
	@if [[ ! -d "$(SRR_DIR)/obj/$(SRR_ARCH)" ]] ; then mkdir "$(SRR_DIR)/obj/$(SRR_ARCH)" ; fi
endif

	@echo 'SRR directory: '$(SRR_DIR)
	@echo 'SRR architecture: '$(SRR_ARCH)
	@echo 'SRR obj files in: '$(SRR_DIR)'/obj/'$(SRR_ARCH)
	@echo 'SRR binary files in: '$(SRR_DIR)'/bin/'$(SRR_ARCH)
	@echo
	@echo 'In order to make an SRR OpenCL code enter its directory and type make'
	@echo '(current and future directories/codes: OpenCL_info, OpenCL_Hello, vec_add,  '
	@echo '                                       mat_transp, mat_vec, mat_mult)'
	@echo

#Export variables in bash
ifeq ($(SHELL),/bin/bash)
	@echo 'To export environment variables, perform the following commands:'
	@echo "export SRR_DIR=\"$(SRR_DIR)\""
	@echo "export SRR_ARCH=\"$(SRR_ARCH)\""
endif


