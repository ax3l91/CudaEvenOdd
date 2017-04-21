#include <iostream>
#include "definitions.h"
#include "ArrayClass.h"
#include "Utils.h"


int range;
ArrayClass *matrix_ptr,*cuda_ptr,*cpu_ptr, *thrust_ptr;
int* cudaMat_ptr, *cpuMat_ptr,*thrustMat_ptr;
char menu = 'a';



int main() {
	//This will attempt to open a connection with the Cuda Enabled GPU
	//Must be called in the beginning because it creates a big (~2sec)
	//overhead when calling the first cuda command
	initializeCuda();

	systemLog("Please input the range of a one Dimensional Matrix");
	std::cin >> range;

	matrix_ptr = new ArrayClass(range, false);
	
	systemLog("Random matrix generated:");
	int* matrix = (*matrix_ptr).getArray();

	//sort with cuda
	systemLog("Sorting with CUDA...");
	startTiming();
	(*matrix_ptr).sort(GPU, &cudaMat_ptr);
	endTiming(true, "GPU");

	//Sort with Thrust
	systemLog("Sorting with THRUST...");
	startTiming();
	(*matrix_ptr).sort(THRUST, &thrustMat_ptr);
	endTiming(true, "Thrust");

	//sort in cpu-sequential
	systemLog("Sorting with CPU...");
	startTiming();
	(*matrix_ptr).sort(CPU, &cpuMat_ptr);
	endTiming(true, "CPU");

	cuda_ptr = new ArrayClass(cudaMat_ptr, range);
	cpu_ptr = new ArrayClass(cpuMat_ptr, range);
	thrust_ptr = new ArrayClass(thrustMat_ptr, range);

	cuda_ptr->checkSort("GPU");
	cpu_ptr->checkSort("CPU");
	thrust_ptr->checkSort("THRUST");

	//(*matrix_ptr).printArray();
	//(*cuda_ptr).printArray();
	//(*cpu_ptr).printArray();
	
	systemLog("Sortings completed. Check above for errors!");
	
	while (menu != 'q') {
		systemLog("Press q to exit and release memory");
		std::cin >> menu;
	}
	exitCuda();

	delete matrix_ptr;
	delete cuda_ptr;
	delete cpu_ptr;
	return 0;
}

