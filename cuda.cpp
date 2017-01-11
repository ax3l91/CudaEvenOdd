#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <device_functions.h>
#include <cuda.h>

int *cudaSort(int a[], const int arraySize)
{
	auto c = new int[arraySize];
	for (int i = 0; i < arraySize; i++) c[i] = a[i];

	// Add vectors in parallel.
	cudaError_t cudaStatus = sortWithCuda(c, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	return c;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t sortWithCuda(int *c, unsigned int size)
{
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_c, c, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	dim3 thread(ceil(sqrt(size)), ceil(sqrt(size)));
	dim3 thread2(84, 84);
	int threads = 128;
	// Launch a kernel on the GPU with one thread for each element.
	//sortKernel<<<1, thread>>>(dev_c,size);
	for (int i = 0; i < size / 2; i++) {
		evenSort << < ((size / 2) + threads - 1) / threads, threads >> > (dev_c, size);
		oddSort << < ((size / 2) - 1 + threads - 1) / threads, threads >> > (dev_c, size);
	}

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);

	return cudaStatus;
}