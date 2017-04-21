#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <iostream>
#include <stdio.h>
#include "math_functions.h"
#include <math.h>

#define THREADS 128

//Implements Even number of Matrix Checks
__global__ void evenSort(int *c, int range) {
	//allocate 2*THREADS shared memory
	__shared__ int sc[2*THREADS];
	//Index of the Main Matrix
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	//Index pointing to shared Matrix
	int si = threadIdx.x;
	//Transfer from Global to Shared Memory. Each Thread
	//transfers it's own index and the next one.
	sc[2*si] = c[2*i];
	sc[2 * si + 1] = c[2 * i + 1];
	//Make sure all threads in Block have written their values
	//into Shared Matrix
	__syncthreads();
	//Make sure we are not calculating out of Matrix Bounds
	if (i < range/2) {
		if (sc[2 * si] > sc[2 * si + 1]) {
			c[2 * i] = sc[2 * si + 1];
			c[2 * i + 1] = sc[2 * si];
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


//Implements Odd number of Matrix Checks
__global__ void oddSort(int *c, int range) {
	//allocate 2*THREADS shared memory
	__shared__ int sc[2*THREADS];
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int si = threadIdx.x;
	sc[2 * si] = c[2 * i + 1];
	sc[2 * si + 1] = c[2 * i + 2];

	//Make sure every thread has passed 2 elements
	//into shared memory before proceeding!
	__syncthreads();
	
	//Make sure we dont go outside of matrix space
	if (i < range / 2 - 1) {
		if (sc[2 * si] > sc[2 * si + 1]) {
			c[2 * i + 1] = sc[2 * si + 1];
			c[2 * i + 2] = sc[2 * si];
		}
	}
}

//OddSort2 allows us to sort Odd-Element Arrays
__global__ void oddSort2(int *c, int range,bool shift) {
	//Now we need 2*THREADS + 1 to make room for sft
	__shared__ int sc[2*THREADS + 1];
	int sft = (shift) ? 1 : 0;
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int si = threadIdx.x;
	sc[2 * si + sft] = c[2 * i + sft];
	sc[2 * si + 1 + sft] = c[2 * i + 1 + sft];
	__syncthreads();

	//We need to floor range/2 to get the correct
	//number of threads that must run this code
	if (i < (int)floorf(range / 2) ) {
		if (sc[2 * si + sft] > sc[2 * si + 1 + sft]) {
			c[2 * i + sft] = sc[2 * si + 1 + sft];
			c[2 * i + 1 + sft] = sc[2 * si + sft];
		}
	}
}

void oddKernel(int * c, int range)
{
	oddSort << < ((range / 2) - 1 + THREADS - 1) / THREADS, THREADS >> > (c, range);
}
void oddKernel2(int * c, int range, bool shift)
{
	oddSort2 << < ((range / 2) + THREADS - 1) / THREADS, THREADS >> > (c, range,shift);
}
