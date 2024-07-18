#include <cuda_runtime.h>
#include "../include/utils.h"


// 当a是一个指针的时候，*a就是这个指针指向的内存的值
// const含义：只要一个变量前用const来修饰，就意味着该变量里的数据只能被访问，而不能被修改，也就是意味着“只读”（readonly）
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) 
    {
        C[i] = A[i] + B[i];
    }  
}

int main(int argc, char **argv){
    printf("%s 开始...\n", argv[0]);

    // 设置设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("使用设备： %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // 设置向量数据
    int nElem = 1<<24;
    printf("向量大小为 %d\n", nElem);

    // 主机申请内存
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // 主机端初始化数据
    iStart = cpuSecond();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = cpuSecond() - iStart;

    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // 主机端计算结果
    iStart = cpuSecond();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = cpuSecond() - iStart;

    // 设备端申请全局内存
    float *d_A, *d_B, *d_C;
    cudaMalloc((float**)&d_A, nBytes);
    cudaMalloc((float**)&d_B, nBytes);
    cudaMalloc((float**)&d_C, nBytes);

    // 将主机数据传到设备端
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    // 在主机端设置线程块，线程格
    int iLen = 256;
    dim3 block  (iLen);
    dim3 grid   ((nElem+block.x-1)/block.x);

    iStart = cpuSecond();
    sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C, nElem);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("线程设置：<<<%d, %d>>> Time elapsed %f sec\n", grid.x, block.x, iElaps);

    // 复制设备端结果到主机
    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    // 对比设备端和主机端计算结果
    checkResult(hostRef, gpuRef, nElem);
    
    // 释放设备端内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 释放主机端内存
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}