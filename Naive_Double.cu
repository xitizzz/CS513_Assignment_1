#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>

#define POWER 24
#define THREAD 1024

using namespace std;

__global__
void add_kernel(double * d_a, double * d_tmp, long k, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i + k < n) {
		d_tmp[i + k] = d_a[i + k] + d_a[i];
	}
}

__host__
void compute_answers(double * a, double * b, long n) {
	double  *d_a, *d_tmp;

	//Allocate memory on GPU
	cudaMalloc(&d_a, n * sizeof(double));
	cudaMalloc(&d_tmp, n * sizeof(double)); //To hold temporary results

										  //Copy content from CPU to GPU
	cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);

	//Copy content in to temporary array
	cudaMemcpy(d_tmp, d_a, n * sizeof(double), cudaMemcpyDeviceToDevice);

	//First pass
	//Launch kernel log n times 
	for (long p = 0; p <= POWER; p++) {
		add_kernel << <(n + THREAD - 1) / THREAD, THREAD >> > (d_a, d_tmp, 1 << p, n);
		cudaMemcpy(d_a, d_tmp, n * sizeof(double), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(d_tmp, d_a, n * sizeof(double), cudaMemcpyDeviceToDevice);

	//Second pass
	//Launch kernel log n times 
	for (long p = 0; p <= POWER; p++) {
		add_kernel << <(n + THREAD - 1) / THREAD, THREAD >> > (d_a, d_tmp, 1 << p, n);
		cudaMemcpy(d_a, d_tmp, n * sizeof(double), cudaMemcpyDeviceToDevice);
	}

	//Copy results back to CPU
	cudaMemcpy(b, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

	//Free memory on GPU
	cudaFree(d_a);
	cudaFree(d_tmp);
}

__host__
double verify_answers(double *a, double * b, long n) {
	double  *v;

	v = (double *)malloc(n * sizeof(double));

	for (int i = 0; i < n; i++) {
		v[i] = a[i];
	}

	for (int i = 1; i < n; i++) {
		v[i] = v[i] + v[i - 1];
	}

	for (int i = 1; i < n; i++) {
		v[i] = v[i] + v[i - 1];
	}
	double maxError = 0;
	for (int i = 0; i < n; i++) {
		maxError = fmax(maxError, fabs(v[i] - b[i]));
	}
	return maxError/v[n-1];
}

int main() {

	double *a, *b;
	long n = 1 << POWER;

	//Allocate memory on CPU
	a = (double *)malloc(n * sizeof(double));
	b = (double *)malloc(n * sizeof(double));

	//Initialize values
	for (long i = 0; i < n; i++) {
		a[i] = ((double)(rand() % n)) / 100;
	}

	//Compute Answers
	compute_answers(a, b, n);

	//Verify Answers
	cout<<"Error margin: " <<verify_answers(a, b, n)<<endl;

	//Free memory on CPU
	free(a);
	free(b);

	return 0;
}
