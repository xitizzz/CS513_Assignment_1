#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>

#define POWER 25
#define THREAD 1024

using namespace std;

__global__
void add_kernel(long * d_a, long * d_tmp, long k, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i + k < n) {
		d_tmp[i + k] = d_a[i + k] + d_a[i];
	}
}

int main() {
	long *a, *d_a, *d_tmp;
	long n = 1 << POWER;

	//Allocate memory on CPU
	a = (long *)malloc(n * sizeof(long));

	//Initialize values
	for (long i = 0; i < n; i++) {
		a[i] =i+1;
	}

	//Allocate memory on GPU
	cudaMalloc(&d_a, n * sizeof(long));
	cudaMalloc(&d_tmp, n * sizeof(long)); //To hold temporary results

	//Copy content from CPU to GPU
	cudaMemcpy(d_a, a, n * sizeof(long), cudaMemcpyHostToDevice);

	//Copy content in to temporary array
	cudaMemcpy(d_tmp, d_a, n * sizeof(long), cudaMemcpyDeviceToDevice);

	//First pass
	//Launch kernel log n times 
	for (long p = 0; p <= POWER; p++) {
		add_kernel << <(n + THREAD - 1) / THREAD, THREAD >> > (d_a, d_tmp, 1 << p, n);
		cudaMemcpy(d_a, d_tmp, n * sizeof(long), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(d_tmp, d_a, n * sizeof(long), cudaMemcpyDeviceToDevice);
	
	//Second pass
	//Launch kernel log n times 
	for (long p = 0; p <= POWER; p++) {
		add_kernel << <(n + THREAD - 1) / THREAD, THREAD >> > (d_a, d_tmp, 1 << p, n);
		cudaMemcpy(d_a, d_tmp, n * sizeof(long), cudaMemcpyDeviceToDevice);
	}

	//Copy results back to CPU
	cudaMemcpy(a, d_a, n * sizeof(long), cudaMemcpyDeviceToHost);

	//Verify results
	for (long i = 0; i < n; i++) {
		//cout << a[i] << "\t";
		if (a[i] != 2 * (i + 1)) {
			//cout << "Incorrect Result " << i << a[i] << endl;
			//break;
		}
	}

	//Free memory on CPU
	free(a);

	//Free memory on GPU
	cudaFree(d_a);
	cudaFree(d_tmp);

	return 0;
}
