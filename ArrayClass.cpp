#include <iostream>
#include <chrono>
#include <string>

#include "ArrayClass.h"
#include "definitions.h"
#include "cuda.h"

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

void ArrayClass::printMatrix(int mat[], int range) {
	for (int i = 0; i < range; i++) {
		std::cout << i << ":" << "[" << mat[i] << "]" << "  ";
	}
	std::cout << std::endl;
}

template <typename T>
void ArrayClass::swap(T A[], int i) {
	T temp = A[i];
	A[i] = A[i + 1];
	A[i + 1] = temp;
}


template <typename T>
T* ArrayClass::evenodd_sort(T mat[]) {
	auto c = new T[range];

	for (int j = 0; j < range / 2; j++) {
		for (int i = 0; i < range; i += 2) {
			if (mat[i] > mat[i + 1]) swap(mat, i);
		}
		for (int i = 1; i < range - 1; i += 2) {
			if (mat[i] > mat[i + 1]) swap(mat, i);
		}
	}
	c = mat;
	return c;
};

