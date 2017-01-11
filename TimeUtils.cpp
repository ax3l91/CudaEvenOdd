#include <iostream>
#include <chrono>
#include <string>


std::chrono::steady_clock::time_point begin;

void startTiming(){
	begin =	std::chrono::steady_clock::now();
}

void endTiming(bool print,std::string str) {
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	if (print) {
		std::cout << str << " completed in: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	}
}

void endTiming(bool print) {
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	if (print) {
		std::cout << " completed in: " << std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count() << " ms" << std::endl;
	}
}