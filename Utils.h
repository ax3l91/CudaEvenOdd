#pragma once
#ifndef UTILS_H
#define UTILS_H
#endif // !UTILS_H

#include <string>

//Time Utilities for Benchmarking
void startTiming();
void endTiming(bool print,std::string str);
void endTiming(bool print);

//General Utilities for Debugging
void systemLog(std::string str);


