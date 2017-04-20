#include <iostream>
#include <chrono>
#include <string>
#include <device_launch_parameters.h>
#include "cuda_runtime.h"
#include "Utils.h"

std::chrono::steady_clock::time_point begin;

void startTiming(){
	begin =	std::chrono::steady_clock::now();
}

void endTiming(bool print,std::string str) {
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	if (print) {
		std::cout << str << " completed in: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	}
}

void endTiming(bool print) {
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	if (print) {
		std::cout << " completed in: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	}
}


void systemLog(std::string string) {
	std::cout << string << std::endl;
}

void initializeCuda() {
	// Choose which GPU to run on, change this on a multi-GPU system.
	short int GPUID = 0;
	cudaError_t cudaStatus = cudaSetDevice(GPUID);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
}

void exitCuda() {
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}
}
