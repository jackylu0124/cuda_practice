#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void printDeviceProperties() {
	int dev_count;
	cudaGetDeviceCount(&dev_count);
	std::cout << "Total Number of Devices: " << dev_count << std::endl;
	std::cout << std::endl;

	cudaDeviceProp dev_prop;
	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&dev_prop, i);
		std::cout << "Device Number: " << i << std::endl;
		std::cout << "	Max Threads Per Block: " << dev_prop.maxThreadsPerBlock << std::endl;
		std::cout << "	Number of SMs: " << dev_prop.multiProcessorCount << std::endl;
		std::cout << "	Clock Rate: " << dev_prop.clockRate << std::endl;
		std::cout << "	(Note: The combination of the clock rate and the number of SMs provides a good indication of the hardware execution capacity of the device.)" << std::endl;
		std::cout << "	Max Threads Allowed along X Dimension of a Block: " << dev_prop.maxThreadsDim[0] << std::endl;
		std::cout << "	Max Threads Allowed along Y Dimension of a Block: " << dev_prop.maxThreadsDim[1] << std::endl;
		std::cout << "	Max Threads Allowed along Z Dimension of a Block: " << dev_prop.maxThreadsDim[2] << std::endl;
		std::cout << "	Max Blocks Allowed along X Dimension of a Grid: " << dev_prop.maxGridSize[0] << std::endl;
		std::cout << "	Max Blocks Allowed along Y Dimension of a Grid: " << dev_prop.maxGridSize[1] << std::endl;
		std::cout << "	Max Blocks Allowed along Z Dimension of a Grid: " << dev_prop.maxGridSize[2] << std::endl;
		std::cout << std::endl;
	}
}

int main() {
	printDeviceProperties();
	return 0;
}