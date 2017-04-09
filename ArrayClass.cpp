#include <iostream>
#include <chrono>
#include <string>
#include "ArrayClass.h"
#include "definitions.h"
#include "timeutils.h"
#include <thread>

//CUDA
#include <device_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include <thrust/sort.h>

using namespace std;

ArrayClass::ArrayClass(int range,bool random) {
	mat = new int[range];
	(*this).range = range;
	populateArray(mat, range, random);
}

ArrayClass::ArrayClass(int range) {
	ArrayClass(range, false);
}

ArrayClass::ArrayClass(int *mat,int range) {
	(*this).mat = mat;
	(*this).range = range;
}

ArrayClass::~ArrayClass() {
	delete[] mat;
}

void ArrayClass::printArray() {
	printMatrix(mat,range);
}

int* ArrayClass::getArray() {
	return mat;
}

void ArrayClass::checkSort() {
	int error = 0;

	for (int i = 0; i < range-1; i++) {

		if (mat[i] > mat[i + 1]) {
			std::cout << "error in: " << i << "with values" << mat[i] << "and" << mat[i + 1] << std::endl;
			error++;
		}
	}
	std::cout << "check sort has found " << error << " errors in the matrix" << std::endl;
}
void ArrayClass::checkSort(std::string str) {
	int error = 0;

	for (int i = 0; i < range - 1; i++) {

		if (mat[i] > mat[i + 1]) {
			std::cout << "error in: " << i << "with values" << mat[i] << "and" << mat[i + 1] << std::endl;
			error++;
		}
	}
	std::cout << str << " check sort has found " << error << " errors in the matrix!" << std::endl;
}

void ArrayClass::sort(int TYPE)
{
}

int * ArrayClass::sort_ptr(int TYPE)
{
	return nullptr;
}

void ArrayClass::sort(int TYPE, int **mat_ptr)
{
	switch (TYPE)
	{
	case GPU:
		*mat_ptr = cudaSort(mat, range);
		break;
	case CPU:
		*mat_ptr = evenodd_sort(mat);
		break;
	case THRUST:
		*mat_ptr = cudaSort(mat, range, true);
		break;
	default:
		cout << "wrong type" << endl;
		break;
	}
}

void ArrayClass::populateArray(int mat[], int range, bool random) {
	if (random) {
		int seed = time(0);
		srand(seed);
	}
	for (int i = 0; i < range; i++) {
		if (random) {
			mat[i] = rand() % 1000;
		}
		else
		{
			mat[i] = 0;
		}
	}

}

void ArrayClass::matrixCpy(int in[], int out[]) {
	for (int i = 0; i < range; i++) {
		out[i] = in[i];
	}
}

void ArrayClass::printMatrix(int mat[], int range) {
	for (int i = 0; i < range; i++) {
		std::cout << i << ":" << "[" << mat[i] << "]" << "  ";
	}
	std::cout << std::endl;
}


template <typename T>
void ArrayClass::swap(T mat[], int i) {
	T temp = mat[i];
	mat[i] = mat[i + 1];
	mat[i + 1] = temp;
}

void ArrayClass::task(unsigned int maxThreads,int threadId)
{
	while (threadId < maxThreads && threadId <range)
	{
		threadId++;
	}
}
void thread1(unsigned int maxThreads ,int threadId,ArrayClass ac) {
	systemLog("hello from thread "+ std::to_string(threadId));
	ac.task(maxThreads, threadId);
}

template <typename T>
T* ArrayClass::evenodd_sort(T mat[]) {
	//How many cores the systems has
	unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
	std::string s = std::to_string(concurentThreadsSupported);
	systemLog(s +" Threads in this system");

	//FAIL way. Needs something better.
	//for (int itNum = 0; itNum < range / 2; itNum++) {
	//	for (int threadId = 0; threadId < concurentThreadsSupported; threadId++) {
	//		std::thread invokeThread (thread1, concurentThreadsSupported, threadId, *this);
	//		invokeThread.join();
	//	}
	//}


	T* matOut = new T[range];
	matrixCpy(mat, matOut);

	for (int j = 0; j < range/2; j++) {
		for (int i = 0; i < range ; i += 2) {
			if (matOut[i] > matOut[i + 1]) {
				swap(matOut, i);
			}
		}
		for (int i = 1; i < range - 1; i += 2) {
			if (matOut[i] > matOut[i + 1]) {
				swap(matOut, i);
			}
		}
	}

	return matOut;
};

int* ArrayClass::cudaSort(int a[], const int arraySize)
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


int * ArrayClass::cudaSort(int mat[], const int range, bool useThrust)
{
	if (!useThrust)
		return cudaSort(mat, range);
	else
	{
		thrust::sort(mat, mat + range);
	}
	return nullptr;
}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t ArrayClass::sortWithCuda(int *c, unsigned int size)
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

	int threads = 128;
	// Launch a kernel on the GPU with one thread for each element.
	//sortKernel<<<1, thread>>>(dev_c,size);
	for (int i = 0; i < size / 2; i++) {
		evenKernel(dev_c, size);
		oddKernel(dev_c, size);
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


