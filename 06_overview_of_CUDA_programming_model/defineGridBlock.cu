#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv){
    // 定义数据大小
    int nElem = 1024;

    // 设置线程格、线程块
    dim3 block  (1024);
    dim3 grid   ((nElem+block.x-1) / block.x);
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    // 重置block
    block.x = 512;
    grid.x = (nElem+block.x-1) / block.x;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    // 重置block
    block.x = 256;
    grid.x = (nElem+block.x-1) / block.x;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    // 重置block
    block.x = 128;
    grid.x = (nElem+block.x-1) / block.x;
    printf("grid.x %d block.x %d\n", grid.x, block.x);

    // 隐式同步
    cudaDeviceReset();
    return 0;
}