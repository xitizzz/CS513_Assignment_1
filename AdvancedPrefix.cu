#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <time.h>

#define THREAD 1024

using namespace std;

__global__
void add_kernel(long * d_a, long * d_tmp, long k, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	i=(i+1)*(n / (gridDim.x*blockDim.x))-1;
	k=k*(n / (gridDim.x*blockDim.x));
	if (i + k < n) {
		d_tmp[i + k] = d_a[i + k] + d_a[i];
	}
}

__global__
void local_sum(long * d_a, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = i*(n / (gridDim.x*blockDim.x));
	int k = (i+1)*(n / (gridDim.x*blockDim.x));
	for (;j < k-1;j++) {
		d_a[j + 1] = d_a[j + 1] + d_a[j];
	}
}

__global__
void local_add(long * d_a, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i == 0) return;
	int j = i*(n / (gridDim.x*blockDim.x))-1;
	for (int k = 1; k < (n / (gridDim.x*blockDim.x)); k++) {
		d_a[j + k] = d_a[j] + d_a[j + k];
	}
}

int main() {
	
	for (int k = 11; k <= 26; k++)
	{
		long *a, *d_a, *d_tmp;
		long n = 1 << k;
		int thread = 1024, block = 2;
		//Allocate memory on CPU
		a = (long *)malloc(n * sizeof(long));

		//Initialize values
		for (long i = 0; i < n; i++) {
			a[i] =  1;
		}

		//Allocate memory on GPU
		cudaMalloc(&d_a, n * sizeof(long));
		cudaMalloc(&d_tmp, n * sizeof(long)); //To hold temporary results

		//Copy content from CPU to GPU
		cudaMemcpy(d_a, a, n * sizeof(long), cudaMemcpyHostToDevice);

		//Copy content in to temporary array
		cudaMemcpy(d_tmp, d_a, n * sizeof(long), cudaMemcpyDeviceToDevice);

		clock_t begin = clock();
		//First pass
		local_sum << <block, thread >> > (d_a, n);
		cudaMemcpy(d_tmp, d_a, n * sizeof(long), cudaMemcpyDeviceToDevice);
		for (long p = 0; p <=log2l(2*thread)-1; p++) {
			add_kernel << <block, thread >> > (d_a, d_tmp, 1 << p, n);
			cudaMemcpy(d_a, d_tmp, n * sizeof(long), cudaMemcpyDeviceToDevice);
		}
		local_add << <block, thread >> > (d_a, n);

		//Second Pass
		local_sum << <block, thread >> > (d_a, n);
		cudaMemcpy(d_tmp, d_a, n * sizeof(long), cudaMemcpyDeviceToDevice);
		for (long p = 0; p <= log2l(2 * thread) - 1; p++) {
			add_kernel << <block, thread >> > (d_a, d_tmp, 1 << p, n);
			cudaMemcpy(d_a, d_tmp, n * sizeof(long), cudaMemcpyDeviceToDevice);
		}
		local_add << <block, thread >> > (d_a, n);

		cudaDeviceSynchronize();

		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC * 1000;
		cout<<"The running time is " << time_spent << " milliseconds."<<endl;
		//Copy results back to CPU
		cudaMemcpy(a, d_a, n * sizeof(long), cudaMemcpyDeviceToHost);
	
		//Verify results
		long *b;
		b = (long *)malloc(n * sizeof(long));
		//Verify results
		for (int i = 0; i < n; i++)
		{
		    b[i] = 1;
		}
		for (int i = 1; i < n; i++)
		{
		    b[i] = b[i] + b[i - 1];
		}
		for (int i = 1; i < n; i++)
		{
		    b[i] = b[i] + b[i - 1];
		}
		for (int i = 1; i < n; i++)
		{
		    if (a[i] != b[i])
		    {
			cout << "Incorrect Result " << i << " " << a[i] <<" "<< b[i] << endl;
			break;
		    }
		}

		//Free memory on CPU
		free(a);

		//Free memory on GPU
		cudaFree(d_a);
		cudaFree(d_tmp);
	}
	return 0;
}
