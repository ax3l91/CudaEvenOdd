#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <stdio.h>
#include "math_functions.h"
#include <math.h>

#define THREADS 128


__global__ void sortKernel(int *c,int range)
{
	//Declare Shared Memory
	__shared__ int sMat[THREADS];

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	printf("thread x:%d thread y:%d blockDim:%d*%d  equals id:%d c[%d] , sMat[%d] \n", threadIdx.x, threadIdx.y, blockDim.x, blockDim.y, i, c[i], sMat[threadIdx.x]);
	
	//Check that we dont fall outside of Array Values
	if (i < range) {

		//Copy the Matrix from Global to Shared Memory
		//sMat[threadIdx.x] = c[i];

		//sync all threads before continuing
		__syncthreads();

		//printf("thread x:%d thread y:%d blockDim:%d*%d  equals id:%d c[%d] , sMat[%d] \n", threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,i,c[i],sMat[threadIdx.x]);

		//Do range/2 iterations of even and then odd which equals to a total of range iterations until the Matrix is Sorted
		for (int it = 0; it < range/2; it++) {
			
			//Do the Even Iteration only using the Shared Memory
			if (i % 2 == 0) {
				//printf("we have c[%d] = %d and sMat[%d] = %d\n",i,c[i], threadIdx.x,sMat[threadIdx.x]);
				/*if (sMat[threadIdx.x] > sMat[threadIdx.x + 1]) {
					printf("swapping c[%d] / sMat[%d]  and c[%d] / sMat[%d] \n", i,threadIdx.x,i + 1,threadIdx.x + 1);
					int temp = sMat[threadIdx.x];
					sMat[threadIdx.x] = sMat[threadIdx.x + 1];
					sMat[threadIdx.x + 1] = temp;
				}*/
				if (c[i] > c[i + 1]) {
					//printf("swapping c[%d] / sMat[%d]  and c[%d] / sMat[%d] \n", i, threadIdx.x, i + 1, threadIdx.x + 1);
					int temp = c[i];
					c[i] = c[i + 1];
					c[i + 1] = temp;
				}
			}

			//Sync and return from shared to global
			__syncthreads();
			//c[i] = sMat[threadIdx.x];

			//Do the Odd Iteration
			if (i % 2 != 0 && i < range-1) {
				//if (i == 31) printf("thread with id:%d is oustide odd %d %d %d \n", i, c[i], c[i+1], c[i+2]);
				//printf("we have c[%d] = %d and sMat[%d] = %d\n", i, c[i], threadIdx.x, sMat[threadIdx.x]);
				if (c[i] > c[i + 1]) {
					//printf("swapping c[%d] / sMat[%d]  and c[%d] / sMat[%d] \n", i, threadIdx.x, i + 1, threadIdx.x + 1);
					int temp = c[i];
					c[i] = c[i + 1];
					c[i + 1] = temp;
				}
				
			}

			//Sync and return from global to shared
			__syncthreads();
			//sMat[threadIdx.x] = c[i];
		}
		__syncthreads();
	}
	
}


void sharedKernel(int *c, int range) {
	sortKernel << < (range + THREADS -1) / THREADS, THREADS >> > (c, range);
}




__global__ void evenSort(int *c, int arraySize) {
	int i = threadIdx.x +blockIdx.x*blockDim.x;
	
	if (i < arraySize/2) {
		if (c[2 * i] > c[2 * i + 1]) {
			int temp = c[2 * i];
			c[2 * i] = c[2 * i + 1];
			c[2 * i + 1] = temp;
		}
	}
}
void evenKernel(int * c, int range)
{
	evenSort << < ((range / 2) + THREADS - 1) / THREADS, THREADS >> > (c, range);
}



__global__ void oddSort(int *c, int arraySize) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < arraySize / 2 - 1) {
		if (c[2 * i + 1] > c[2 * i + 2]) {
			int temp = c[2 * i + 1];
			c[2 * i + 1] = c[2 * i + 2];
			c[2 * i + 2] = temp;
		}
	}
}
__global__ void oddSort2(int *c, int arraySize,bool shift) {
	int sft = 0;
	if (shift) sft = 1;
	int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i < (int)floorf(arraySize / 2)) {
		if (c[2 * i +sft] > c[2 * i + 1+sft]) {
			int temp = c[2 * i+sft];
			c[2 * i+sft] = c[2 * i + 1+sft];
			c[2 * i + 1+sft] = temp;
		}
	}
}

void oddKernel(int * c, int range)
{
	oddSort << < ((range / 2) - 1 + THREADS - 1) / THREADS, THREADS >> > (c, range);
}
void oddKernel2(int * c, int range, bool shift)
{
	oddSort2 << < ((range / 2) - 1 + THREADS - 1) / THREADS, THREADS >> > (c, range,shift);
}
