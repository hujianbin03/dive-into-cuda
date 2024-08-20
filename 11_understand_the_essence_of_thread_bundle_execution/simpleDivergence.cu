#include <cuda_runtime.h>
#include "../include/utils.h"

__global__ void warmingup(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if((tid/warpSize) % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel1(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if (tid % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel2(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((tid / warpSize) % 2 == 0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    c[tid] = a + b;
}

__global__ void mathKernel3(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    bool ipred = (tid % 2 == 0);
    if (ipred){
        ia = 100.0f;
    }
    if (!ipred){
        ib = 200.0f;
    }

    c[tid] = a + b;
}

__global__ void mathKernel4(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    int itid = tid >> 5;

    if (itid & 0x01 == 0)
    {
        a = 100.0f;
    }
    else
    {
        b = 200.0f;
    }

    c[tid] = a + b;
}

int main(int argc, char **argv){
    // 设置设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("%s 使用设备 %d: %s\n", argv[0], dev, deviceProp.name);

    // 设置数据
    int size = 64;
    int blocksize = 64;
    if(argc > 1) blocksize = atoi(argv[1]);
    if(argc > 2) size      = atoi(argv[2]);
    printf("数据大小： %d\n", size);

    // 设置线程格、块
    dim3 block (blocksize, 1);
    dim3 grid  ((size+block.x-1)/block.x,1);
    printf("内核配置为：(block %d grid %d)\n", block.x, grid.x);

    // gpu申请内存
    float *d_C;
    size_t nBytes = size * sizeof(float);
    cudaMalloc((float**)&d_C, nBytes);

    // 执行warmup消除开销
    double iStart, iElaps;
    cudaDeviceSynchronize();
    iStart = seconds();
    warmingup<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("warmup          <<< %4d %4d >>> 消耗时间 %lf sec\n", grid.x, block.x, iElaps);

    // 执行mathkernel1
    iStart = seconds();
    mathKernel1<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathkernel1     <<< %4d %4d >>> 消耗时间 %lf sec\n", grid.x, block.x, iElaps);

    // 执行mathkernel2
    iStart = seconds();
    mathKernel2<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel2     <<< %4d %4d >>> 消耗时间 %lf sec\n", grid.x, block.x, iElaps);

     // 执行mathkernel3
    iStart = seconds();
    mathKernel3<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel3     <<< %4d %4d >>> 消耗时间 %lf sec\n", grid.x, block.x, iElaps);

    // 执行mathkernel4
    iStart = seconds();
    mathKernel4<<<grid, block>>>(d_C);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    printf("mathKernel4     <<< %4d %4d >>> 消耗时间 %lf sec\n", grid.x, block.x, iElaps);

    //释放内存
    cudaFree(d_C);
    cudaDeviceReset();
    return EXIT_SUCCESS;    
}a