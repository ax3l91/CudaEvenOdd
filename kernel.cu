#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <stdio.h>
#include "math_functions.h"
#include <math.h>

#define THREADS 128

//Implements Even number of Matrix Checks
__global__ void evenSort(int *c, int arraySize) {
	int i = threadIdx.x +blockIdx.x*blockDim.x;
	
	//Make sure we are not calculating out of Matrix Bounds
	if (i < arraySize/2) {
		if (c[2 * i] > c[2 * i + 1]) {
			int temp = c[2 * i];
			c[2 * i] = c[2 * i + 1];
			c[2 * i + 1] = temp;
		}
	}
}
/*We will Spawn a ceil(range/2) number of blocks,
each containing THREADS number of threads. Each block
will sort 2*THREADS elements.
*/
void evenKernel(int * c, int range)
{
	evenSort << < ((range / 2) + THREADS - 1) / THREADS, THREADS >> > (c, range);
}



__global__ void oddSort(int *c, int arraySize) {
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	//Make sure we are not calculating out of Matrix Bounds
	if (i < arraySize / 2 - 1) {
		if (c[2 * i + 1] > c[2 * i + 2]) {
			int temp = c[2 * i + 1];
			c[2 * i + 1] = c[2 * i + 2];
			c[2 * i + 2] = temp;
		}
	}
}

//OddSort2 allows us to sort Odd-Element Arrays
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
