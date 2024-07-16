#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdio.h>

// 当a是一个指针的时候，*a就是这个指针指向的内存的值
// const含义：只要一个变量前用const来修饰，就意味着该变量里的数据只能被访问，而不能被修改，也就是意味着“只读”（readonly）
void sumArraysOnHost(float *A, float *B, float *C, const int N){
    for (int idx=0; idx<N; idx++){
        C[idx] = A[idx] + B[idx];
    }
}

void initialData(float *ip, int size){
    // 为随机数生成不同的种子
    // time_t是一个数据类型，用于表示时间
    time_t t;
    // &t获取变量t的地址
    srand((unsigned int) time(&t));

    for (int i=0; i<size; i++){
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(int argc, char **argv){
    int nElem = 1024;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);
    printf("h_A: %f + h_B: %f = h_C: %f\n", *h_A, *h_B, *h_C);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}























