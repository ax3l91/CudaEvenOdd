#ifndef ARRAY_CLASS_H
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
	void sort(int TYPE);
	void sort(int TYPE, int **mat_ptr);
	int* sort_ptr(int TYPE);

	
private:
	template<typename T>
	void swap(T A[], int i);
	template<typename T>
	T* evenodd_sort(T mat[]);
	void printMatrix(int mat[], int range);
	void populateArray(int mat[], int range, bool random);
	int *mat;
	int range;
};


#endif // !ARRAY_CLASS_H
