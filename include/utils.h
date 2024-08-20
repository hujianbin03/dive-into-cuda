#ifndef _UTILS_H
#define _UTILS_H
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

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef _WIN32
#	include <windows.h>
#else
#	include <sys/time.h>
#endif

void initialData(float *ip, int size){
    // 为随机数生成不同的种子
    // time_t是一个数据类型，用于表示时间
    time_t t;
    // &t获取变量t的地址
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float)(rand()&0xff) / 10.0f;
    }
}

void initialData_int(int* ip, int size){
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i<size; i++)
	{
		ip[i] = int(rand()&0xff);
	}
}

void initDevice(int devNum){
  int dev = devNum;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp,dev));
  printf("使用设备： %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
}

// The best computing power
void getTheBestComputerPowDevice(){
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 1){
        int maxMultiprocessors = 0, maxDevice = 0;
        for (int device=0; device<numDevices; device++){
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, device);
            if (maxMultiprocessors < props.multiProcessorCount){
                maxMultiprocessors = props.multiProcessorCount;
                maxDevice = device;
            }
        }
        printf("计算能力最优的gpu编号: %d\n", maxDevice);
        cudaSetDevice(maxDevice);
    }
    else{
        printf("设备只有一个gpu: %d\n", numDevices);
        cudaSetDevice(numDevices);
    }
}

double cpuSecond(){
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
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

void printMatrix(float * C,const int nx,const int ny)
{ 
    float *ic=C;
    printf("Matrix<%d,%d>:\n",ny,nx);
    for(int i=0;i<ny;i++)
    {
        for(int j=0;j<nx;j++)
        {
        printf("%6f ",ic[j]);
        }
    ic+=nx;
    printf("\n");
  }
}

void printMatrix_int(int * C,const int nx,const int ny)
{
    int *ic=C;
    printf("Matrix<%d,%d>:\n",ny,nx);
    for(int i=0;i<ny;i++)
    {
        for(int j=0;j<nx;j++)
        {
        printf("%3d ",ic[j]);
    }
    ic+=nx;
    printf("\n");
    }  
}

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

#endif  //_UTILS_H