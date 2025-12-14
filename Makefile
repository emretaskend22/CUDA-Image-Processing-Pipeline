# Compiler settings
CC = gcc
NVCC ?= nvcc
CFLAGS = -O3 -std=c11 -Wall
NVCCFLAGS = -O3 -arch=sm_75

# Target executable
TARGET = cuda_image_processing

# Source files
HOST_SOURCE = host_code.cu
CUDA_SOURCE = cuda_kernels.cu
HEADER = image_utils.h

# Default target: build CUDA version (required)
all: $(HOST_SOURCE) $(CUDA_SOURCE) $(HEADER)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA $(HOST_SOURCE) $(CUDA_SOURCE) -o $(TARGET)
	@echo "Built CUDA version: $(TARGET)"

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o output_*.ppm

# Phony targets
.PHONY: all clean

