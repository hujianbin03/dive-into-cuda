#include <stdio.h>

__global__ void helloFromGPU(void)
{
    printf("hello world from gpu!\n")
}

int main(void)
{
    printf("hello world from gpu111\n");
    helloFromGPU <<<1, 10>>>;
    cudaDeviceReset();
    return 0
}