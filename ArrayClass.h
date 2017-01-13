#ifndef ARRAY_CLASS_H
#include "cuda_runtime.h"
#include <string>
#define ARRAY_CLASS_H

class ArrayClass
{
public:
	ArrayClass(int range,bool random);
	ArrayClass(int range);
	ArrayClass(int* mat,int range);
	~ArrayClass();
	void printArray();
	int* getArray();
	void checkSort();
	void checkSort(std::string type);
	void sort(int TYPE);
	void sort(int TYPE, int **mat_ptr);
	int* sort_ptr(int TYPE);

	
private:
	template<typename T>
	void swap(T mat[], int i);
	template<typename T>
	T* evenodd_sort(T mat[]);
	void printMatrix(int mat[], int range);
	void populateArray(int mat[], int range, bool random);
	void matrixCpy(int in[], int out[]);
	int* cudaSort(int mat[],const int range);
	cudaError_t sortWithCuda(int *mat,unsigned int range);


	//object properties
	int *mat;
	int range;
};


#endif // !ARRAY_CLASS_H
