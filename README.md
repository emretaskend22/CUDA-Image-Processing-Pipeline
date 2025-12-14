# CUDA Optimized Image Processing Pipeline

## Project Overview
This repository contains an optimized implementation of a two-stage image processing pipeline using **CUDA**. The project focuses on applying various GPU optimization techniques to achieve significant performance improvements over a naive approach.

### Pipeline Stages
1.  **Grayscale Conversion:** Converts an input RGB image to a grayscale image using the standard luminosity formula: $Y = 0.299 \times R + 0.587 \times G + 0.114 \times B$.
2.  **5x5 Gaussian Blur:** Applies a 5x5 Gaussian blur filter to the resulting grayscale image.

The codebase includes several iterations of the pipeline, demonstrating progressive optimization from a baseline implementation to advanced techniques like kernel fusion, shared memory usage, and persistent kernels.

## Requirements
* **NVIDIA GPU:** A CUDA-enabled graphics card (The `Makefile` is configured for `sm_75`, but this can be adjusted).
* **CUDA Toolkit:** Version 10.0 or newer.
* **GCC/G++:** C/C++ compiler for host code.

## Build and Run

### 1. Building the Project
The project can be built using the provided `Makefile`.

```bash
# Build the main CUDA executable
make all
