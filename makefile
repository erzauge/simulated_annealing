CC_CPU		=g++
CC_GPU		=nvcc
CFLAGS_CPU	=-std=c++11 -fopenmp -Wall -Wextra -O2
CFLAGS_GPU	=-Xptxas -dlcm=ca -lineinfo -Wno-deprecated-gpu-targets -std=c++11 -O2
LIBS_CPU	= -lm
LIBS_GPU	= -lcurand
INC_CPU		=-Itclap/include
INC_GPU		=-Icub/cub -Itclap/include
SRC 		=src
SRC_TEST	=./test
RELESE_PATH	= obj/relese/
DEBUG_PATH	= obj/debug/
OPJ_PATH	= $(RELESE_PATH)

ifneq ($(DEBUG),)
	# CC_CPU = clang++
	CFLAGS_CPU	+= -fsanitize=address -g -fno-omit-frame-pointer -D DEBUG
	CFLAGS_GPU	+= -g -G -D DEBUG
	OPJ_PATH	= $(DEBUG_PATH)
endif


NAME_CPU	= annealing_cpu
NAME_GPU	= annealing_gpu

CPU_TEST_SOURCES	:=$(wildcard $(SRC_TEST)/*.cpp)
GPU_TEST_SOURCES	:=$(wildcard $(SRC_TEST)/*.cu)

CPU_SOURCES	:=$(wildcard $(SRC)/*.cpp)
GPU_SOURCES	:=$(wildcard $(SRC)/*.cu)

GPU_TEST_SOURCES :=$(wildcard $(SRC_TEST)/*.cu)
CPU_TEST_SOURCES :=$(wildcard $(SRC_TEST)/*.cpp)

CPU_OBJS	:= $(CPU_SOURCES:$(SRC)/%.cpp=$(OPJ_PATH)%.o)
GPU_OBJS	:= $(GPU_SOURCES:$(SRC)/%.cu=$(OPJ_PATH)%.obj)


INC_TEST_GPU = $(INC_GPU)  $(CUSTEMINC) $(GTEST_INC) -I$(SRC) -I$(SRC_TEST)

DEPS_CPU	:= $(CPU_OBJS:%.o=%.d)
DEPS_GPU	:= $(GPU_OBJS:%.obj=%.d)

.PRECIOUS: $(DEPS_GPU)

VERSION	= $(shell git describe --long --dirty)

all: $(NAME_CPU) $(NAME_GPU)

debug_cpu:
	make $(NAME_CPU) DEBUG=0

debug_gpu:
	make $(NAME_PU) DEBUG=0

# .PHONY: convert
# convert: obj/relese/Logging.o obj/relese/sys_file.o
# 	make -C convert
# 	ln -sf convert/convert convert_gs
#
# .PHONY: plaquette
# plaquette: obj/relese/Logging.o obj/relese/sys_file.o
# 	make -C plaquette
# 	ln -sf plaquette/plaquette plaquette_sum


$(NAME_CPU): $(CPU_OBJS)
	$(CC_CPU) $(LIBS_CPU) $(CFLAGS_CPU) $^ -o $@

$(NAME_GPU): $(GPU_OBJS) $(RELESE_PATH)sys_file.o $(RELESE_PATH)Logging.o $(RELESE_PATH)block.o $(RELESE_PATH)random_str.o $(RELESE_PATH)bin_io.o
	$(CC_GPU) $(CFLAGS_GPU) $(LIBS_GPU) $^ -o $@



-include $(DEPS_CPU)
-include $(DEPS_GPU)


$(OPJ_PATH)%.o: $(SRC)/%.cpp | $(OPJ_PATH)
	$(CC_CPU) $(VFLAG) $(CFLAGS_CPU) $(INC_CPU) -MP -MMD -c $< -o $@

$(OPJ_PATH)%.obj: $(SRC)/%.cu $(OPJ_PATH)%.d | $(OPJ_PATH)
	$(CC_GPU) $(VFLAG) $(CFLAGS_GPU) $(INC_GPU) -c $< -o $@

$(OPJ_PATH)%.d: $(SRC)/%.cu | $(OPJ_PATH)
	$(CC_GPU) $(VFLAG) $(CFLAGS_GPU) $(INC_GPU) -M $< -MT $(@:%.d=%.obj) -o $@

test/$(OPJ_PATH)%.obj: $(SRC_TEST)/%.cu test/$(OPJ_PATH)%.d | test/$(OPJ_PATH)
	$(CC_GPU) $(VFLAG) $(CFLAGS_GPU) $(INC_TEST_GPU) -c $< -o $@

test/$(OPJ_PATH)%.d: $(SRC_TEST)/%.cu | test/$(OPJ_PATH)
	$(CC_GPU) $(VFLAG) $(CFLAGS_GPU) $(INC_TEST_GPU) -M $< -MT $(@:%.d=%.obj) -o $@

$(OPJ_PATH):
	mkdir -p $@

test/$(OPJ_PATH):
	mkdir -p $@

.PHONY: $(SRC)/version.h
$(SRC)/version.h:
	@echo update version.h
	echo "#define VERSION \"$(VERSION)\"" >$(SRC)/version.h

.PHONY: clean
clean:
	rm -rf $(OPJ_PATH) test/$(OPJ_PATH) $(DEBUG_PATH) test/$(DEBUG_PATH) $(NAME_CPU) $(NAME_GPU)  utest convert_gs
	make -C convert clean

.PHONY: clean_debug
clean_debug:
	rm -r $(DEBUG_PATH) test/$(DEBUG_PATH)
