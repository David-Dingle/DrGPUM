# Modified from https://github.com/Jokeren/compute-sanitizer-samples/tree/master/MemoryTracker
PROJECT ?= gpu-patch.fatbin gpu-patch-address.fatbin gpu-patch-aux.fatbin gpu-patch-torch-aux.fatbin gpu-patch-aux-torchview.fatbin
PROJECT_ANALYSIS ?= gpu-analysis.fatbin

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda
SANITIZER_PATH ?= $(CUDA_PATH)/compute-sanitizer
CUPTI_PATH ?= $(CUDA_PATH)

NVCC := $(CUDA_PATH)/bin/nvcc

INCLUDE_DIRS := -I$(CUDA_PATH)/include -I$(SANITIZER_PATH)/include -I$(CUPTI_PATH)/include -Iinclude
SRC_DIR := src
CXXFLAGS := $(INCLUDE_DIRS) -O3 --fatbin

ARCHS := 70 72 75 80 86

# Generate SASS code for each SM architectures
$(foreach sm,$(ARCHS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(ARCHS)))
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)

all: $(PROJECT) $(PROJECT_ANALYSIS)

ifdef PREFIX
install: all
endif

$(PROJECT): %.fatbin : $(SRC_DIR)/%.cu
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) --compile-as-tools-patch -o $@ -c $<

$(PROJECT_ANALYSIS): %.fatbin : $(SRC_DIR)/%.cu
	$(NVCC) $(CXXFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

ifdef PREFIX
install: $(PROJECT) $(PROJECT_ANALYSIS)
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/bin
	cp -rf $(PROJECT) $(PROJECT_ANALYSIS) $(PREFIX)/lib
	cp -rf include $(PREFIX)
endif

clean:
	rm -f $(PROJECT) $(PROJECT_ANALYSIS)
