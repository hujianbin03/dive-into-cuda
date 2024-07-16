#include <cuda_runtime.h>
#include <stdio.h>

__global__ void checkIndex(void){
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
        gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
        blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
        gridDim.x,gridDim.y,gridDim.z);
}

int main(int argc, char **argv){
    // 定义数据大小
    int nElem = 6;

    // 定义线程格和线程块
    dim3 block  (3);
    dim3 grid   ((nElem+block.x-1) / block.x);

    // 检查主机端：线程格和线程块的维度
    printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);

    // 检查设备端：线程格和线程块的维度
    checkIndex <<<grid, block>>>();

    // 隐式同步CPU和GPU
    cudaDeviceReset();
    return 0;
}

