#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

float* readImage(int* width, int* height, std::string path) {
	std::ifstream reader;
	reader.open(path);
	if (!reader.is_open()) {
		std::cerr << "Unable to open file." << std::endl;
		exit(1);
	}

	// Read file content
	// Read magic number
	std::string magicNumber;
	reader >> magicNumber;
	if (magicNumber != "P3") {
		std::cerr << "The magic number is not P3 (The file is not an ASCII PPM file)." << std::endl;
		exit(1);
	}

	// Read width and height
	std::string widthStr, heightStr;
	reader >> widthStr >> heightStr;
	*width = stoi(widthStr);
	*height = stoi(heightStr);
	std::cout << "The image dimension is " << *width << " x " << *height << "." << std::endl;

	// Read range
	std::string rangeStr;
	float range;
	reader >> rangeStr;
	range = stof(rangeStr);

	// Read pixel values of the image, allocate a vector, and store the image's pixel values in the vector
	int n = *width * *height * 3;
	float* vec = (float*) malloc(n * sizeof(float));
	if (vec == NULL) {
		std::cout << "Malloc for vector in readImage(int* width, int* height, std::string path) fails." << std::endl;
		exit(1);
	}
	std::string pixelValStr;
	for (int i = 0; i < n; i++) {
		reader >> pixelValStr;
		vec[i] = stof(pixelValStr);
	}

	// Close the file
	reader.close();

	return vec;
}

void writeImage(float* vec, int width, int height, std::string path) {
	std::ofstream writer;
	writer.open(path);
	if (!writer.is_open()) {
		std::cerr << "Unable to write file." << std::endl;
		exit(1);
	}

	writer << "P3\n";
	writer << width << " " << height << "\n";
	writer << "255\n";
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			for (int c = 0; c < 3; c++) {
				int index = (row * width + col) * 3 + c;
				writer << (int) vec[index] << " ";
			}
		}
		writer << "\n";
	}

	writer.close();
}

void colorToGrayCPU(std::string readPath, std::string writePath) {
	int width, height;
	float* inVec = readImage(&width, &height, readPath);
	float* outVec = (float*) malloc(width * height * 3 * sizeof(float));
	if (outVec == NULL) {
		std::cout << "Malloc for vector in colorToGrayCPU(std::string readPath, std::string writePath) fails." << std::endl;
		exit(1);
	}
	
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int index = (row * width + col) * 3;
			float r = inVec[index];
			float g = inVec[index + 1];
			float b = inVec[index + 2];

			float gray = 0.21 * r + 0.72 * g + 0.07 * b;
			outVec[index] = gray;
			outVec[index + 1] = gray;
			outVec[index + 2] = gray;
		}
	}
	writeImage(outVec, width, height, writePath);

	// Free allocated memories
	free(outVec);
	free(inVec);
}

int main() {
	// TODO
	std::string readPath = "pineapple_pizza.ppm";
	std::string writePath = "pineapple_pizza_gray_cpu.ppm";
	colorToGrayCPU(readPath, writePath);
	return 0;
}