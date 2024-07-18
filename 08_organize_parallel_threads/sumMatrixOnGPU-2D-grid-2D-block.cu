#include <cuda_runtime.h>
#include "../include/utils.h"

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny){
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny){
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

int main(int argc, char **argv){
    printf("%s 开始...\n", argv[0]);

    // 设置设备
    initDevice(0);

    // 设置矩阵维度
    int nx = 1 <<14;
    int ny = 1 <<14;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("矩阵大小为: nx %d ny %d\n", nx, ny);

    // 主机申请内存
    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // 初始化矩阵
    double iStart = cpuSecond();
    initialData(h_A, nxy);
    initialData(h_B, nxy);
    double iElaps = cpuSecond() - iStart;

    // 初始化内存
    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // 主机计算矩阵相加
    iStart = cpuSecond();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = cpuSecond() - iStart;

    // 设备申请内存
    float *d_MatA, *d_MatB, *d_MatC;
    cudaMalloc((void **)&d_MatA, nBytes);
    cudaMalloc((void **)&d_MatB, nBytes);
    cudaMalloc((void **)&d_MatC, nBytes);

    // 将数据从主机端传输到设备端
    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice);

    // 设置线程块，线程格
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);

    // 执行核函数
    iStart = cpuSecond();
    sumMatrixOnGPU2D <<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>> Time elapsed %f sec\n",
        grid.x,grid.y,block.x,block.y,iElaps); 

    // 将设备端结果传到主机端
    cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost);

    // 对比主机和设备结果
    checkResult(hostRef, gpuRef, nxy);

    // 释放内存
    cudaFree(d_MatA);
    cudaFree(d_MatB);
    cudaFree(d_MatC);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    cudaDeviceReset();
    return 0;
} 