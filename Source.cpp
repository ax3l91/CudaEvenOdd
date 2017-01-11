#include <iostream>

#include "definitions.h"
#include "ArrayClass.h"
#include "timeutils.h"



int range = 8;
ArrayClass *matrix_ptr,*cuda_ptr,*cpu_ptr;
int* cudaMat_ptr, *cpuMat_ptr;



int main() {
	matrix_ptr = new ArrayClass(range, true);
	//(*matrix_ptr).printArray();
	std::cout << "Random matrix generated:" << std::endl;
	int* matrix = (*matrix_ptr).getArray();

	//sort with cuda
	startTiming();
	(*matrix_ptr).sort(GPU, &cudaMat_ptr);
	endTiming(true, "GPU");

	//sort in cpu-sequential
	startTiming();
	(*matrix_ptr).sort(CPU, &cpuMat_ptr);
	endTiming(true, "CPU");

	cuda_ptr = new ArrayClass(cudaMat_ptr, range);
	cuda_ptr->checkSort();
	//(*cuda_ptr).printArray();
	cpu_ptr = new ArrayClass(cpuMat_ptr, range);
	cpu_ptr->checkSort();
	//(*cpu_ptr).printArray();
	

	system("pause");
	return 0;
}

