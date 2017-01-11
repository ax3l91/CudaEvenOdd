#include <iostream>
#include <random>
#include <time.h>
#include <chrono>


#include "kernel.cuh"


void checkSort(int mat[]);

int range;
void printMatrix(int mat[]);

template <typename T>
void swap(T A[],int i) {
	T temp = A[i];
	A[i] = A[i+1];
	A[i+1] = temp;
};

template <typename T>
void evenodd_sort(T mat[]) {

	for (int j = 0; j < range/2; j++) {
		for (int i = 0; i < range; i += 2) {
			if (mat[i] > mat[i + 1]) swap(mat, i);
			//printMatrix(mat);
		}
		for (int i = 1; i < range-1; i += 2) {
			if (mat[i] > mat[i + 1]) swap(mat, i);
			//printMatrix(mat);
		}
	}
};



int main() {
	int seed = time(0);
	srand(seed);
	int initial = 5000;

	//dimension of the array
	range = 50000;
	auto matrix = new int[range];

	for (int i = 0; i < range; i++) {
		matrix[i] = rand()%1000;
	}
	std::cout << "Random matrix generated:" << std::endl;
	//printMatrix(matrix);

	//sort with cuda
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	auto cudaMatrix = cudaSort(matrix, range);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Cuda completed in: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() <<" ms"<< std::endl;
	//printMatrix(cudaMatrix);
	checkSort(cudaMatrix);

	//sort in cpu-sequential
	std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();
	evenodd_sort(matrix);
	std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	std::cout << "CPU completed in: " << std::chrono::duration_cast<std::chrono::milliseconds> (end2 - begin2).count() << " ms" << std::endl;
	//printMatrix(matrix);
	checkSort(matrix);
	

	system("pause");
	return 0;
}


void printMatrix(int mat[]) {
	for (int i = 0; i < range; i++) {
		std::cout << i << ":" << "[" << mat[i] << "]" << "  ";
	}
	std::cout << std::endl;
}

void checkSort(int mat[]) {
	int error = 0;

	for (int i = 0; i < range-1; i++) {
		
		if (mat[i] > mat[i + 1]) {
			std::cout << "error in: " << i << "with values" << mat[i]<< "and" <<mat[i+1]<< std::endl;
			error++;
		}
	}

	std::cout << "check sort has found " << error << " errors in the matrix" << std::endl;
}
