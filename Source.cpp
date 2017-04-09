#include <iostream>

#include "definitions.h"
#include "ArrayClass.h"
#include "timeutils.h"


int range = 50000;
ArrayClass *matrix_ptr,*cuda_ptr,*cpu_ptr;
int* cudaMat_ptr, *cpuMat_ptr,*thrustMat_ptr;
char menu = 'a';



int main() {
	matrix_ptr = new ArrayClass(range, true);
	
	std::cout << "Random matrix generated..." << std::endl;
	int* matrix = (*matrix_ptr).getArray();

	//sort with cuda
	std::cout << "Sorting with CUDA..." << std::endl;
	startTiming();
	(*matrix_ptr).sort(GPU, &cudaMat_ptr);
	endTiming(true, "GPU");

	//Sort with Thrust
	std::cout << "Sorting with Thrust..." << std::endl;
	startTiming();
	(*matrix_ptr).sort(THRUST, &thrustMat_ptr);
	endTiming(true, "Thrust");

	//sort in cpu-sequential
	std::cout << "Sorting with CPU..." << std::endl;
	startTiming();
	(*matrix_ptr).sort(CPU, &cpuMat_ptr);
	endTiming(true, "CPU");

	cuda_ptr = new ArrayClass(cudaMat_ptr, range);
	cpu_ptr = new ArrayClass(cpuMat_ptr, range);

	cuda_ptr->checkSort("GPU");
	cpu_ptr->checkSort("CPU");

	//(*matrix_ptr).printArray();
	//(*cuda_ptr).printArray();
	//(*cpu_ptr).printArray();
	
	std::cout << "Sortings completed. Check above for errors!" << std::endl;
	
	while (menu != 'q') {
		std::cout << "press q to exit and release memory" << std::endl;
		std::cin >> menu;
	}

	delete matrix_ptr;
	delete cuda_ptr;
	delete cpu_ptr;
	return 0;
}

