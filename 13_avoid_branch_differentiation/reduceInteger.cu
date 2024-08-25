#include "../include/utils.h"
#include <cuda_runtime.h>

int recursiveReduce(int *data, int const size)
{
    if(size == 1)   return data[0];

    int const stride = size / 2;
    for(int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

__global__ void warmup(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局数据指针转换为此线程块的loacl指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx > n) return;

    // 全局内存就地归约
    for(int stride = 1; stride < blockDim.x; stride *=2)
    {
        if((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // 块内同步
        __syncthreads();
    }
    // 将此块的结果写入全局内存
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

/**
* @param g_idata    本地内存数组： 存放每个线程块的部分和
* @param g_odata    全局内存数组： 用来存放整个数组
*/
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局数据指针转换为此线程块的loacl指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx > n) return;

    // 全局内存就地归约
    for(int stride = 1; stride < blockDim.x; stride *=2)
    {
        if((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // 块内同步
        __syncthreads();
    }
    // 将此块的结果写入全局内存
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局数据指针转换为此线程块的loacl指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx > n) return;

    // 全局内存就地归约
    for(int stride = 1; stride < blockDim.x; stride *=2)
    {
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }
        __syncthreads();
    }
    // 将此块的结果写入全局内存
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局数据指针转换为此线程块的loacl指针
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if(idx > n) return;

    // >>= 1: 相当于 /= 2
    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if(tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // 将此块的结果写入全局内存
    if(tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

int main(int argc, char **argv)
{
    // 设置设备
    initDevice(0);

    bool bResult = false;

    // 初始化
    int size = 1<<24;   // 要归约的元素总数
    printf("   数组大小： %d    ", size);

    // 设置内核配置
    int blocksize = 512;    // 初始化线程块大小
    if(argc > 1)
    {
        blocksize = atoi(argv[1]);  // 可以通过命令行设置线程块大小
    }
    dim3 block  (blocksize, 1);
    dim3 grid   ((size+block.x-1)/block.x, 1);
    printf("线程格 %d 线程块 %d\n", grid.x, block.x);

    // 申请主机内存
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp = (int *) malloc(bytes);
    
    // 初始化数组
    for(int i = 0; i < size; i++)
    {
        h_idata[i] = (int)(rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // 申请设备端内存
    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata, bytes);
    cudaMalloc((void **) &d_odata, grid.x * sizeof(int));

    // cpu 进行归约
    iStart = seconds();
    int cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      花费 %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

    // 内核1： 预热归约
    // 因为第一次启动gpu会慢一些，可能导致性能差距，所以需要预热，跑一下gpu, 实际运行的warmup和reduceNeighbored是一样的代码
    // 但是warmup会慢一些
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    warmup<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu warmup      花费 %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // 内核1： 归约
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu Neighbored  花费 %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // 内核2： 归约减少分化
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu Neighbored2 花费 %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // 内核3： 交错配对归约
    cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    iStart = seconds();
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
    cudaDeviceSynchronize();
    iElaps = seconds() - iStart;
    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;
    for(int i = 0; i < grid.x; i++)
    {
        gpu_sum += h_odata[i];
    }
    printf("gpu Interleaved 花费 %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // 释放主机内存
    free(h_idata);
    free(h_odata);

    // 释放设备内存
    cudaFree(d_idata);
    cudaFree(d_odata);

    // 重置设备
    cudaDeviceReset();

    // 检查结果
    bResult = (gpu_sum == cpu_sum);
    if(!bResult)
    {
        printf("测试失败!\n");
    }
    return EXIT_SUCCESS;
}