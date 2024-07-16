#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess)                                                   \
    {                                                                           \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                           \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));     \
        exit(1);                                                                \
    }                                                                           \
} 

void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8 ;
    int match = 1;
    for (int i=0; i<N; i++){
        if (fabsf(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
            break;
        }
    }
    if (match){
        printf("Arrays match.\n\n");
        return;
    }
}

void initialData(float *ip, int size){
    // 为随机数生成不同的种子
    // time_t是一个数据类型，用于表示时间
    time_t t;
    // &t获取变量t的地址
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

// 当a是一个指针的时候，*a就是这个指针指向的内存的值
// const含义：只要一个变量前用const来修饰，就意味着该变量里的数据只能被访问，而不能被修改，也就是意味着“只读”（readonly）
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnGPU(float*a,float*b,float*res)
{
  int i=threadIdx.x;
  res[i]=a[i]+b[i];
}

int main(int argc, char **argv){
    printf("%s 开始...\n", argv[0]);

    // 设置设备
    int dev = 0;
    cudaSetDevice(dev);

    // 设置向量数据
    int nElem = 32;
    printf("向量大小为 %d\n", nElem);

    // 主机申请内存
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    // 主机端初始化数据
    initialData(h_A, nElem);
    initialData(h_B, nElem);

    memset(hostRef, 0, nBytes);
    memset(gpuRef,  0, nBytes);

    // 设备端申请全局内存
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // 将主机数据传到设备端
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));

    // 在主机端设置线程块，线程格
    dim3 block  (nElem);
    dim3 grid   (nElem/block.x);

    sumArraysOnGPU<<< grid, block >>>(d_A, d_B, d_C);
    printf("线程设置：<<<%d, %d>>>\n", grid.x, block.x);

    // 复制设备端结果到主机
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // 主机端计算结果
    sumArraysOnHost(h_A, h_B, hostRef, nElem);

    // 对比设备端和主机端计算结果
    checkResult(hostRef, gpuRef, nElem);

    printf("打印数组A，数组B，主机结果，设备结果前5个数：\n");
    for(int i=0; i < 5; i++){
        printf("h_A[%d]=%f h_B[%d]=%f hostRef[%d]=%f gpuRef[%d]=%f\n", i, h_A[i], i, h_B[i], i, *hostRef[i], i, *gpuRef[i]);
    }

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