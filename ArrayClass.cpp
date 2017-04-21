#include <iostream>
#include <chrono>
#include <string>
#include "ArrayClass.h"
#include "definitions.h"
#include "Utils.h"
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
			printf("Error in m[%d] = %d and m[%d] = %d \n", i, mat[i], i + 1, mat[i + 1]);
			error++;
		}
	}
	std::cout << "check sort has found " << error << " errors in the matrix" << std::endl;
}
void ArrayClass::checkSort(std::string str) {
	int error = 0;

	for (int i = 0; i < range - 1; i++) {

		if (mat[i] > mat[i + 1]) {
			printf("Error in m[%d] = %d and m[%d] = %d \n", i, mat[i], i + 1, mat[i + 1]);
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
	/*Find out What kind of Sort to Run and return a Pointer to 
	a new sorted Matrix*/
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
			//Make a matrix that is ascending to test every iteration
			mat[range- i -1] = 2*i + 10;
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

//TODO: Parallelize exploiting all CPU cores.
template <typename T>
T* ArrayClass::evenodd_sort(T mat[]) {
	//How many cores the systems has
	unsigned int concurentThreadsSupported = std::thread::hardware_concurrency();
	std::string s = std::to_string(concurentThreadsSupported);
	systemLog(s +" Threads in this system");

	//FAIL way. Needs something better.
	/*for (int itNum = 0; itNum < range / 2; itNum++) {
		for (int threadId = 0; threadId < concurentThreadsSupported; threadId++) {
			std::thread invokeThread (thread1, concurentThreadsSupported, threadId, *this);
			invokeThread.join();
		}
	}
	*/

	T* matOut = new T[range];
	matrixCpy(mat, matOut);

	//Sequencial Single-Threaded sorting.
	//Matrix has Even number of Elements
	if (range % 2 == 0) {
		for (int j = 0; j < range / 2; j++) {
			for (int i = 0; i < range; i += 2) {
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
	}
	//Matrix has Odd number of Elements
	else {
		for (int j = 0; j < range / 2; j++) {
			for (int i = 0; i < range -1; i += 2) {
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
	}
	return matOut;
};

int* ArrayClass::cudaSort(int a[], const int arraySize)
{
	auto c = new int[arraySize];
	for (int i = 0; i < arraySize; i++) c[i] = a[i];

	//Sort using Cuda, pass the matrix and range
	cudaError_t cudaStatus = sortWithCuda(c, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed! \n");
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
		return mat;
	}
	return nullptr;
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t ArrayClass::sortWithCuda(int *c, unsigned int size)
{
	int *dev_c = 0;
	cudaError_t cudaStatus;

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
	endTiming(true, "GPU buffer Allocation ");

	//Check if range is Even or Odd
	if (size % 2 == 0) {
		//Matrix has an even number of elements
		for (int i = 0; i < size / 2; i++) {
			evenKernel(dev_c, size);
			oddKernel(dev_c, size);
		}
	}
	else
	{
		//Matrix has an odd number of elements
		for (int i = 0; i < (int) floorf(size / 2) + 1; i++) {
			oddKernel2(dev_c, size,false);
			oddKernel2(dev_c, size, true);
		}
	}
	endTiming(true, "Kernel Work ");

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	any errors encountered during the launch.*/
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
		goto Error;
	}
	endTiming(true, "Device Synchronize ");


	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	endTiming(true, "Copy from Device to Host ");

Error:
	cudaFree(dev_c);

	return cudaStatus;
}


