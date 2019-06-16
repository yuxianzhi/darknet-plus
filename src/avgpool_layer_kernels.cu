#include <hip/hip_runtime.h>
#include "rocrand/rocrand.h"
#include "rocblas.h"

#include "avgpool_layer.h"
#include "cuda.h"

__global__ void forward_avgpool_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    output[out_index] = 0;
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        output[out_index] += input[in_index];
    }
    output[out_index] /= w*h;
}

__global__ void backward_avgpool_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index = (k + c*b);
    for(i = 0; i < w*h; ++i){
        int in_index = i + h*w*(k + b*c);
        in_delta[in_index] += out_delta[out_index] / (w*h);
    }
}

FUNC_OP void forward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    hipLaunchKernelGGL((forward_avgpool_layer_kernel), dim3(hip_gridsize(n)), dim3(BLOCK), 0, 0, n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    check_error(hipPeekAtLastError());
}

FUNC_OP void backward_avgpool_layer_gpu(avgpool_layer layer, network net)
{
    size_t n = layer.c*layer.batch;

    hipLaunchKernelGGL((backward_avgpool_layer_kernel), dim3(hip_gridsize(n)), dim3(BLOCK), 0, 0, n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(hipPeekAtLastError());
}

