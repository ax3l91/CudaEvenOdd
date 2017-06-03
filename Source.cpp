#include <iostream>
#include "definitions.h"
#include "ArrayClass.h"
#include "Utils.h"


int range, iter;
ArrayClass *matrix_ptr,*cuda_ptr,*cpu_ptr, *thrust_ptr;
int* cudaMat_ptr, *cpuMat_ptr,*thrustMat_ptr;
int cudatime, cputime, thrusttime;
char menu = 'a', ver = 'a';
bool verbose;



int main() {
	//This will attempt to open a connection with the Cuda Enabled GPU
	//Must be called in the beginning because it creates a big (~2sec)
	//overhead when calling the first cuda command
	initializeCuda();
	fileIOInit();

	systemLog("Please input the starting range of a one Dimensional Matrix");
	std::cin >> range;
	systemLog("Input number of iterations. Each iteration is range*2");
	std::cin >> iter;
	systemLog("Be verbose? Y/n");
	std::cin >> ver;
	while (ver!='y' && ver!= 'n')
	{
		systemLog("Y/n please");
		std::cin >> ver;
	}
	if (ver == 'y') verbose = true;
	else verbose = false;

	for (int i = 0; i < iter; i++) {
		matrix_ptr = new ArrayClass(range, false);

		systemLog(verbose,"Random matrix generated:");
		int* matrix = (*matrix_ptr).getArray();

		//sort with cuda
		systemLog(verbose,"Sorting with CUDA...");
		startTiming();
		(*matrix_ptr).sort(verbose,GPU, &cudaMat_ptr);
		cudatime = endTiming(verbose, "GPU");

		//Sort with Thrust
		systemLog(verbose,"Sorting with THRUST...");
		startTiming();
		(*matrix_ptr).sort(verbose,THRUST, &thrustMat_ptr);
		thrusttime = endTiming(verbose, "Thrust");

		//sort in cpu-sequential
		systemLog(verbose,"Sorting with CPU...");
		startTiming();
		(*matrix_ptr).sort(verbose,CPU, &cpuMat_ptr);
		cputime = endTiming(verbose, "CPU");

		cuda_ptr = new ArrayClass(cudaMat_ptr, range);
		cpu_ptr = new ArrayClass(cpuMat_ptr, range);
		thrust_ptr = new ArrayClass(thrustMat_ptr, range);

		cuda_ptr->checkSort(verbose,"GPU");
		cpu_ptr->checkSort(verbose,"CPU");
		thrust_ptr->checkSort(verbose,"THRUST");

		//(*matrix_ptr).printArray();
		//(*cuda_ptr).printArray();
		//(*cpu_ptr).printArray();

		systemLog(verbose,"Sortings completed. Check above for errors!");

		fileIO(range, cputime, cudatime, thrusttime);
		range = range*2;
		delete matrix_ptr;
		delete cuda_ptr;
		delete cpu_ptr;
	}

	fileIOClose();

	while (menu != 'q') {
		systemLog("You can find the csv named exported_times to compare the benchmarks \nPress q to exit \n ");
		std::cin >> menu;
	}
	exitCuda();

	return 0;
}

