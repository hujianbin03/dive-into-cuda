#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/utils.h"

int main(int argc, char **argv){
    printf("%s 开始...\n", argv[0]);

    getTheBestComputerPowDevice();
}