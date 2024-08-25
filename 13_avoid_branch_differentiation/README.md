&emsp;&emsp;有时，控制流依赖于线程索引。线程束中的条件执行可能引起线程束分化，这会导致内核性能变差。通过重新组织数据的获取模式，可以减少或避免线程束分化。在本节里，将会以并行归约为例，介绍避免分支分化的基本技术。
## 1.并行归约问题
假设要对一个有N个元素的整数数组求和。使用如下的串行代码很容易实现算法：
```c
int sum = 0;
for(int i = 0; i < N; i++)
{
    sum += array[i];
}
```

&emsp;&emsp;如果有大量的数据元素会怎么样呢？如何通过并行计算快速求和呢？鉴于加法的结合律和交换律，数组元素可以以任何顺序求和。所以可以用以下的方法执行并行加法运算：
1. 将输入向量划分到更小的数据块中。
2. 用一个线程计算一个数据块的部分和。
3. 对每个数据块的部分和再求和得出最终结果。

&emsp;&emsp;并行加法的一个常用方法是使用迭代成对实现。把向量的数据分成对，然后用不同线程计算每一对元素，得到的结果作为输入继续分成对，迭代的进行，直到最后一个元素。
成对的划分常见的方法有以下两种：
* 相邻配对：元素与它们直接相邻的元素配对![img.png](..%2Fasset%2F13%2Fimg.png)
* 交错配对：根据给定的跨度配对元素![img_1.png](..%2Fasset%2F13%2Fimg_1.png)
下面用C语言实现一个递归交错配对方法，注意：下面这段代码是没有考虑数组长度非2的整数幂次的情况。
```c
int recursiveReduce(int *data, int const size)
{
    if(size == 1)   return data[0];

    int const stride = side / 2;
    for(int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride)
}
```
&emsp;&emsp;分析上述代码，可以看出其满足加法的交换律和结合律。其他如最大值、最小值、平均值和乘法也可以替换加法。**在向量中执行满足交换律和结合律的运算，被称为归约问题**。并行归约问题是这种运算的并行执行。归约是一种常见的计算方式，归约的归有递归的意思，约就是减少，即每次迭代计算方式都是相同的（归），从一组多个数据最后得到一个数（约）。
## 2.并行归约中的分化
&emsp;&emsp;现在我们来写相邻配对的cuda代码，下图所示的是相邻配对方法的内核实现流程。每个线程将相邻的两个元素相加产生部分和。
&emsp;&emsp;在这个内核里，有两个全局内存数组：一个大数组用来存放整个数组，进行归约；另一个小数组用来存放每个线程块的部分和。每个线程块在数组的一部分上独立地执行操作。循环中迭代一次执行一个归约步骤。归约是在就地完成的，这意味着在每一步，全局内存里的值都被部分和替代。
![img_2.png](..%2Fasset%2F13%2Fimg_2.png)
具体代码如下：
```c
/**
* @param g_idata    本地内存数组： 存放每个线程块的部分和
* @param g_odata    全局内存数组： 用来存放整个数组
*/
__global__ void reduceNeighbored(int *g_idata, int, *g_odata, unsigned int n)
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
```

&emsp;&emsp;这里面唯一要注意的地方就是同步指令，__syncthreads语句可以保证，线程块中的任一线程在进入下一次迭代之前，在当前迭代里每个线程的所有部分和都被保存在了全局内存中。进入下一次迭代的所有线程都使用上一步产生的数值。在最后一个循环以后，整个线程块的和被保存进全局内存中。
```c
__syncthreads();
```
&emsp;&emsp;两个相邻元素间的距离被称为跨度，也就是变量stride，初始化均为1。在每一次归约循环结束后，这个间隔就被乘以2。在第一次循环结束后，idata（全局数据指针）的偶数元素将会被部分和替代。在第二次循环结束后，idata的每四个元素将会被新产生的部分和替代。因为线程块间无法同步，所以每个线程块产生的部分和被复制回了主机，并且在那儿进行串行求和，如下图所示。
![img_3.png](..%2Fasset%2F13%2Fimg_3.png)
完整代码：[**https://github.com/dive-into-cuda**](https://github.com/hujianbin03/dive-into-cuda)  
这里只列出了主函数：
```c
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
    printf("cpu 归约     花费 %lf ms cpu_sum: %d\n", iElaps, cpu_sum);

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
    printf("gpu 预热     花费 %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

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
    printf("gpu 相邻归约 花费 %lf ms gpu_sum: %d <<<grid %d block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

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
```
运行结果如下：
![img_4.png](..%2Fasset%2F13%2Fimg_4.png)
## 3.改善并行归约的分化
接下来我们会一步一步发现问题并优化问题。观察上节的核函数reduceNeighbored，并注意以下条件表达式：
```c
if((tid % (2 * stride)) == 0)
```
&emsp;&emsp;因为上述语句只对偶数ID的线程为true，所以这会导致很高的线程束分化。在并行归约的第一次迭代中，只有ID为偶数的线程执行这个条件语句的主体，在第二次迭代中，只有四分之一的线程是活跃的，我们希望所有的线程都被调度执行，这样才能发挥最佳的性能。**通过重新组织每个线程的数组索引来强制ID相邻的线程执行求和操作**，线程束分化就能被归约了。下图展示了这种实现。和上节的实现图相比，部分和的存储位置并没有改变，但是工作线程已经更新了。
![img_5.png](..%2Fasset%2F13%2Fimg_5.png)
具体代码如下：
```c
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
```
注意内核中的下述语句，它为每个线程设置数组访问索引：
```c
int index = 2 * stride * tid;
```
因为跨度乘以了2，所以下面的语句使用线程块的前半部分来执行求和操作：
```c
if (index < blockDim.x)
```
&emsp;&emsp;对于一个有512个线程的块来说，前8个线程束执行第一轮归约，剩下8个线程束什么也不做。在第二轮里，前4个线程束执行归约，剩下12个线程束什么也不做，可以看下表。因此，这样就彻底不存在分化了。在最后五轮中，当每一轮的线程总数小于线程束的大小时，分化就会出现。在下一节将会介绍如何处理这一问题。
```c
// blockdim.x = 512

// 第一次迭代： tid：0～255，即256个线程，256/32=8个线程束
// t:0 s:1 i:0
// t:1 s:1 i:2
// t:2 s:1 i:4
// t:3 s:1 i:6
// ...
// t:255 s:1 i:510
// t:256 s:1 i:512

// 第二次迭代： tid：0～127，即128个线程，128/32=4个线程束
// t:0 s:2 i:4
// ...
// t:127 s:2 i:508  
// t:128 s:2 i:512

// s = 4
// s = 8
// s = 16
// s = 32 
// s = 64
// s = 128

// 最后一次迭代: tid: 0, 即一个线程，1/32=1个线程束
// t:0 s:256 i:0
// t:1 s:256 i:512
```
现在在主函数中加入reduceNeighboredLess的调用，如下：
```c
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
```
执行结果如下：
![img_6.png](..%2Fasset%2F13%2Fimg_6.png)
可以通过测试不同的指标来解释这两个内核之间的不同行为。
* 用inst_per_warp指标来查看每个线程束上执行指令数量的平均值。
  * ```c
       nvprof --metrics inst_per_warp ./reduceInteger
       ```
  * 结果总结如下，原来的内核在每个线程束里执行的指令数是新内核的两倍多，它是原来实现高分化的一个指示器：![img_7.png](..%2Fasset%2F13%2Fimg_7.png)
* 用gld_throughput指标来查看内存加载吞吐量：
  *  ```c
        nvprof --metrics gld_throughput ./reduceInteger
     ```
  * 结果总结如下，新的实现拥有更高的加载吞吐量，因为虽然I/O操作数量相同，但是其耗时更短：![img_8.png](..%2Fasset%2F13%2Fimg_8.png)
## 4.交错配对的归约
&emsp;&emsp;与相邻配对方法相比，交错配对方法颠倒了元素的跨度。初始跨度是线程块大小的一半，然后在每次迭代中减少一半（如下图所示）。在每次循环中，每个线程对两个被当前跨度隔开的元素进行求和，以产生一个部分和。与相邻配对实现图相比，交错归约的工作线程没有变化。但是，**每个线程在全局内存中的加载/存储位置是不同的**。
![img_9.png](..%2Fasset%2F13%2Fimg_9.png)
交错归约的内核代码如下所示：
```c
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
```

注意核函数中的下述语句，两个元素间的跨度被初始化为线程块大小的一半，然后在每次循环中减少一半：
```c
for(int stride = blockDim.x / 2; stride > 0; stride >>= 1)
```
下面的语句在第一次迭代时强制线程块中的前半部分线程执行求和操作，第二次迭代时是线程块的前四分之一，以此类推：
```c
if(tid < stride)
```
下面的代码增加到主函数中，执行交错归约的代码：
```c
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

```
执行结果如下：
![img_10.png](..%2Fasset%2F13%2Fimg_10.png)
&emsp;&emsp;交错实现比Neighbored实现快了1.99倍，比Neighbored2实现快了1.16倍。这种性能的提升主要是由reduceInterleaved函数里的**全局内存加载/存储模式导致的**。在后面的章节里会介绍更多有关于全局内存加载/存储模式对内核性能的影响。reduceInterleaved函数和reduceNeighboredLess函数依然存在相同的线程束分化。