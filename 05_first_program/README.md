&emsp;&emsp;在本节，你将开始编写在GPU上运行的第一个内核代码。像其他任何编程语言一样编写GPU上的第一个程序是输出字符串“Hello World”。设置好的环境，以及打开你常用的编辑器一起开始吧。现在你要准备好写你的第一个CUDA C程序。写一个CUDA C程序，你需要以下几个步骤：
1. 用专用扩展名.cu来创建一个源文件。
2. 使用CUDA nvcc编译器来编译程序。
3. 从命令行运行可执行文件，这个文件有可在GPU上运行的内核代码。

## 1. Hello World

首先，我们编写一个C语言程序来输出“Hello World”，如下所示：
```c
#include <stdio.h>
int main(void)
{
    printf("Hello World from CPU!\n");
}
```

>在 C 语言中，stdio.h：标准输入输出库，是最常用和最基本的库之一，它提供了一组函数，用于处理输入和输出操作，包括读取和写入字符、字符串、格式化输出和文件操作等。

&emsp;&emsp;把代码保存到hello.cu中，然后使用nvcc编译器来编译。CUDA nvcc编译器和gcc编译
器及其他编译器有相似的语义。
```linux
nvcc hello.cu -o hello
```
&emsp;&emsp;然后在同目录下，就会生成hello可执行文件，点击执行，就会看到输出“Hello World from CPU”。

## 2. 内核函数
> 代码仓库：[https://github.com/hujianbin03/dive-into-cuda](https://github.com/hujianbin03/dive-into-cuda)

接下来，编写一个内核函数，命名为helloFromGPU，用它来输出字符串“Hello World
from GPU！”。
```c
__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU!\n");
}
```
修改main函数，以下是全部代码：
```c
#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("Hello World from GPU!\n");
}

int main(void)
{
    printf("Hello World from CPU!\n");
    helloFromGPU <<<1, 10>>>();
    cudaDeviceReset();
    return 0;
}	
```
简单介绍其中几个关键字和写法：
* 修饰符__global__告诉编译器这个函数将会从CPU中调用，然后在GPU上执行。
	```c
	__global__
	```
* 这句话C语言中没有’<<<>>>’，是CUDA扩展出来的部分。三重尖括号意味着从主线程到设备端代码的调用。一个内核函数通过一组线程来执行，所有线程执行相同的代码。三重尖括号里面的参数是执行配置，用来说明使用**多少线程来执行内核函数**。在这个例子中，有10个GPU线程被调用。
	```c
	hello_world<<<1,10>>>();
	```
* 如果没有cudaDeviceReset();，就不会打印出“Hello World from GPU!”，因为这句话包含了隐式同步，**GPU和CPU执行程序是异步的，核函数调用后成立刻会到主机线程继续，而不管GPU端核函数是否执行完毕**，所以上面的程序就是GPU刚开始执行，CPU已经退出程序了，所以我们要等GPU执行完了，再退出主机线程。
	```c
	cudaDeviceReset();
	```

一个典型的CUDA编程结构包括5个主要步骤。
1. 分配GPU内存。
2. 从CPU内存中拷贝数据到GPU内存。
3. 调用CUDA内核函数来完成程序指定的运算。
4. 将数据从GPU拷回CPU内存。
5. 释放GPU内存空间。

上面的hello world只到第三步，没有内存交换。

## 3. 使用CUDAC编程难吗
&emsp;&emsp;CPU编程和GPU编程的主要区别是程序员对GPU架构的熟悉程度。要写好程序，需要用并行思维进行思考并对GPU架构有了基本的了解。例如，数据局部性在并行编程中是一个非常重要的概念。数据局部性指的是数据重用，以降低内存访问的延迟。数据局部性有两种基本类型：
* 时间局部性：是指在相对较短的时间段内数据和/或资源的重用。
* 空间局部性：是指在相对较接近的存储空间内数据元素的重用。

CUDA中有内存层次和线程层次的概念，使用如下结构，有助于你对线程执行进行更高层次的控制和调度：
* 内存层次结构
* 线程层次结构

&emsp;&emsp;当用CUDA C编写程序时，实际上你只编写了**被单个线程调用的一小段串行代码**。GPU处理这个内核函数，然后通过启动成千上万个线程来实现并行化，所有的线程都执行相同的计算。CUDA编程模型提供了一个层次化地组织线程的方法，它直接影响到线程在GPU上的执行顺序。

CUDA抽象了硬件细节，且不需要将应用程序映射到传统图形API上。CUDA核中有3个关键抽象：
* 线程组的层次结构
* 内存的层次结构
* 障碍同步

&emsp;&emsp;实际上，CUDA平台已经为程序员做了很多底层、框架等工作，且生态会越来越完善，我们的目标应是**学习GPU架构的基础及掌握CUDA开发工具和环境**。NVIDIA为C和C++开发人员提供了综合的开发环境以创建GPU加速应用程序，当你熟悉这些工具的使用之后，你会发现使用CUDA C语言进行编程是非常简单高效的，具体包括以下几种（**当然这是一本2017年的书，相关的工具可能有更新，请关注NVIDIA官网！**）：
* NVIDIA Nsight集成开发环境
* CUDA-GDB命令行调试器
* 用于性能分析的可视化和命令行分析器
* CUDA-MEMCHECK内存分析器
* GPU设备管理工具

## 4. 总结
&emsp;&emsp;随着计算机架构和并行编程模型的发展，逐渐有了现在所用的异构系统。CPU＋GPU的异构系统在高性能计算领域已经成为主流。这种变化使并行设计范例有了根本性转变：在GPU上执行数据并行工作，而在CPU上执行串行和任务并行工作。而CUDA平台帮助提高了异构架构的性能和程序员的工作效率。目前看起来，在异构系统中编写一个具有成百上千个核的CUDA程序就像编写一个串行程序那样简单，哈哈哈，继续加油！