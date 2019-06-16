#ifndef CUDA_H
#define CUDA_H

#define FUNC_OP

#include "darknet.h"

#ifdef GPU

void check_error(hipError_t status);
rocblas_handle blas_handle();
int *hip_make_int_array(int *x, size_t n);
void hip_random(float *x_gpu, size_t n);
float hip_compare(float *x_gpu, float *x, size_t n, char *s);
dim3 hip_gridsize(size_t n);

#ifdef CUDNN
cudnnHandle_t cudnn_handle();
#endif

#endif
#endif
