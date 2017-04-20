#pragma once
#ifndef KERNEL_CUH
#define KERNEL_CUH

#endif // !KERNEL_CUH

void sortKernel(int *c, int range);
void sharedKernel(int * c, int range);
void evenKernel(int *c, int range);
void oddKernel(int *c, int range);