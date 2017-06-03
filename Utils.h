#pragma once
#ifndef UTILS_H
#define UTILS_H
#endif // !UTILS_H

#include <string>
#include "cuda.h"

//Time Utilities for Benchmarking
void startTiming();
int endTiming(bool print,std::string str);
void endTiming(bool print);

//General Utilities for Debugging
void systemLog(bool verbose,std::string str);
void systemLog(std::string str);

//Cuda utilities
void initializeCuda();
void exitCuda();

//File IO
void fileIO(int range, int cpu, int cuda, int thrust);
void fileIOInit();
void fileIOClose();



