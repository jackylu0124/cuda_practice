#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>

// Always remember to add these 3 header files
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Create and allocate space for a random vector of size n where each element is in the range of 0-49
int* genRandVec(int n) {
	int* vec = (int*) malloc(n * sizeof(int));
	if (vec == NULL) {
		printf("Malloc for vector in genRandVec(int n) fails.\n");
		exit(1);
	}
	for (int i = 0; i < n; i++) {
		vec[i] = rand() % 50;
	}
	return vec;
}

// Create and allocate space for an empty vector of size n
int* genEmptyVec(int n) {
	int* vec = (int*) malloc(n * sizeof(int));
	if (vec == NULL) {
		printf("Malloc for vector in genEmptyVec(int n) fails.\n");
		exit(1);
	}
	return vec;
}

void printVec(int* vec, int n) {
	for (int i = 0; i < n; i++) {
		printf("Element %d is : %d\n", i, vec[i]);
	}
}

__global__
void vecAddKernel(int* v1, int* v2, int* v3, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n) {
		v3[i] = v1[i] + v2[i];
	}
}

void vecAdd(int* v1, int* v2, int* v3, int n) {
	int size = n * sizeof(int);
	int* d_v1, *d_v2, *d_v3;

	cudaMalloc((void**) &d_v1, size);
	cudaMemcpy(d_v1, v1, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_v2, size);
	cudaMemcpy(d_v2, v2, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**) &d_v3, size);
	cudaMemcpy(d_v3, v3, size, cudaMemcpyHostToDevice);

	// Lauch kernel here
	vecAddKernel <<<ceil(n / 256.0), 256>>>(d_v1, d_v2, d_v3, n);

	cudaMemcpy(v3, d_v3, size, cudaMemcpyDeviceToHost);

	cudaFree(d_v1);
	cudaFree(d_v2);
	cudaFree(d_v3);
}

int main() {
	int length = 5;
	int* v1 = genRandVec(length);
	int* v2 = genRandVec(length);
	int* v3 = genEmptyVec(length);

	vecAdd(v1, v2, v3, length);

	// Print out content of v1
	printf("Elements of v1 are: \n");
	printVec(v1, length);
	printf("\n");

	// Print out content of v2
	printf("Elements of v2 are: \n");
	printVec(v2, length);
	printf("\n");

	// Print out content of v3
	printf("Elements of v3 are: \n");
	printVec(v3, length);
	printf("\n");

	free(v1);
	free(v2);
	free(v3);

	return 0;
}