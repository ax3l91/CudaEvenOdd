
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <stdio.h>

#define threads 128


__global__ void sortKernel(int *c,int arraySize)
{
	
	int i = threadIdx.x + threadIdx.y*blockDim.x;
	
	if (i < arraySize) {
		printf("thread x:%d thread y:%d blockDim:%d*%d  equals id:%d \n", threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,i);
		//__syncthreads();
		for (int it = 0; it < arraySize/2; it++) {

			if (i % 2 == 0) {

				//if (i == 2) printf("thread with id:%d is oustide even %d %d %d \n", i,c[i-1],c[i],c[i+1]);
				if (c[i] > c[i + 1]) {
					int temp = c[i];
					c[i] = c[i + 1];
					c[i + 1] = temp;
				}
				
			}
			__syncthreads();

			if (i % 2 != 0 && i <arraySize-1) {
				//if (i == 31) printf("thread with id:%d is oustide odd %d %d %d \n", i, c[i], c[i+1], c[i+2]);
				if (c[i] > c[i + 1]) {
					int temp = c[i];
					c[i] = c[i + 1];
					c[i + 1] = temp;
				}
				
			}
			__syncthreads();
		}
		__syncthreads();
	}
	
}







__global__ void evenSort(int *c, int arraySize) {
	int i = threadIdx.x +blockIdx.x*blockDim.x;
	
	if (i < arraySize/2) {
		//printf("even thread x:%d blockDim:%d*%d blockidx.x:%d  equals id:%d \n", threadIdx.x, blockDim.x, blockDim.y, blockIdx.x, i);
		if (c[2 * i] > c[2 * i + 1]) {
			int temp = c[2 * i];
			c[2 * i] = c[2 * i + 1];
			c[2 * i + 1] = temp;
		}
	}
	__syncthreads();
}
void evenKernel(int * c, int range)
{
	evenSort << < ((range / 2) + threads - 1) / threads, threads >> > (c, range);
}



__global__ void oddSort(int *c, int arraySize) {
	int i = threadIdx.x +blockIdx.x*blockDim.x;
	
	if (i < arraySize/2 -1) {
		//printf("odd thread x:%d blockDim:%d*%d blockidx.x:%d  equals id:%d \n", threadIdx.x, blockDim.x, blockDim.y, blockIdx.x, i);
		if (c[2 * i+1] > c[2 * i + 2]) {
			int temp = c[2 * i+1];
			c[2 * i+1] = c[2 * i + 2];
			c[2 * i +2] = temp;
		}
	}
	__syncthreads();
}
void oddKernel(int * c, int range)
{
	oddSort << < ((range / 2) - 1 + threads - 1) / threads, threads >> > (c, range);
}
