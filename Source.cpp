#include <iostream>

#include "definitions.h"
#include "ArrayClass.h"
#include "timeutils.h"


int range = 12;
ArrayClass *matrix_ptr,*cuda_ptr,*cpu_ptr;
int* cudaMat_ptr, *cpuMat_ptr;



int main() {
	matrix_ptr = new ArrayClass(range, true);
	(*matrix_ptr).printArray();
	std::cout << "Random matrix generated:" << std::endl;
	int* matrix = (*matrix_ptr).getArray();

	//sort with cuda
	std::cout << "Sorting with CUDA..." << std::endl;
	startTiming();
	(*matrix_ptr).sort(GPU, &cudaMat_ptr);
	endTiming(true, "GPU");

	//sort in cpu-sequential
	std::cout << "Sorting with CPU..." << std::endl;
	startTiming();
	(*matrix_ptr).sort(CPU, &cpuMat_ptr);
	endTiming(true, "CPU");

	cuda_ptr = new ArrayClass(cudaMat_ptr, range);
	cpu_ptr = new ArrayClass(cpuMat_ptr, range);

	matrix_ptr->printArray();
	cuda_ptr->checkSort("GPU");
	(*cuda_ptr).printArray();
	cpu_ptr->checkSort("CPU");
	(*cpu_ptr).printArray();
	
	std::cout << "Sortings completed. Check above for errors!" << std::endl;
	std::cout << "Press any key to end program...";
	system("pause");

	delete matrix_ptr;
	delete cuda_ptr;
	return 0;
}

