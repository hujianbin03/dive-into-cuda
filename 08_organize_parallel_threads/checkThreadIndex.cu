#include <cuda_runtime.h>
#include "../include/utils.h"

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf("thread_id(%d,%d) block_id(%d,%d) coordinate(%d,%d)"
          "global index %2d ival %2d\n",threadIdx.x,threadIdx.y,
          blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);
}

int main(int argc, char **argv){
    printf("%s 开始...\n", argv[0]);

    // 设置设备
    initDevice(0);

    // 设置矩阵维度
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    // 主机申请内存
    int *h_A;
    h_A = (int *)malloc(nBytes);

    // 初始化矩阵
    initialData_int(h_A, nxy);
    printMatrix_int(h_A, nx, ny);

    // 设备申请内存
    int *d_MatA;
    cudaMalloc((void **)&d_MatA, nBytes);

    // 将数据从主机端传输到设备端
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    // 设置线程块，线程格
    dim3 block(4, 2);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // 执行核函数
    printThreadIndex <<<grid, block>>>(d_MatA, nx, ny);
    
    // 隐式同步
    cudaDeviceSynchronize();

    // 释放内存
    cudaFree(d_MatA);
    free(h_A);

    cudaDeviceReset();
    return 0;
} 