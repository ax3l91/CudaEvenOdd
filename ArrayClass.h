#ifndef ARRAY_CLASS_H
#include "cuda_runtime.h"
#include <string>
#define ARRAY_CLASS_H

class ArrayClass
{
public:
	//Constructors
	ArrayClass(int range,bool random);
	ArrayClass(int range);
	ArrayClass(int* mat,int range);
	//Destructor
	~ArrayClass();
	//Public functions provide an API to private ones
	void printArray();
	int* getArray();
	void checkSort();
	void checkSort(bool verbose,std::string type);
	void sort(int TYPE);
	//void sort(int TYPE, int **mat_ptr);
	void sort(bool verbose,int TYPE, int **mat_ptr);
	int* sort_ptr(int TYPE);
	void task(unsigned int maxThreads, int threadId);

	
private:
	template<typename T>
	void swap(T mat[], int i);
	template<typename T>
	T* evenodd_sort(T mat[]);
	void printMatrix(int mat[], int range);
	void populateArray(int mat[], int range, bool random);
	void matrixCpy(int in[], int out[]);
	int* cudaSort(bool verbose,int mat[],const int range);
	int* cudaSort(bool verbose,int mat[], const int range,bool useThrust);
	cudaError_t sortWithCuda(bool verbose,int *mat,unsigned int range);


	//object properties
	int *mat;
	int range;
};


#endif // !ARRAY_CLASS_H


