/*
WILLL NOT WORK ON NON POWER OF TWO, STILL WORKING
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <ctime>
#include <time.h> //for clock_gettime
#include <stdio.h> //for printf

#define POWER 28

using namespace std;

__global__
void add_kernel(double * d_a, double * d_tmp, long k, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	i=(i+1)*(n / (gridDim.x*blockDim.x))-1;
	k=k*(n / (gridDim.x*blockDim.x));
	if (i + k < n) {
		d_tmp[i + k] = d_a[i + k] + d_a[i];
	}
}

__global__
void local_sum(double * d_a, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = i*(n / (gridDim.x*blockDim.x));
	int k = (i+1)*(n / (gridDim.x*blockDim.x));
	for (;j < k-1;j++) {
		d_a[j + 1] = d_a[j + 1] + d_a[j];
	}
}

__global__
void local_add(double * d_a, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i == 0) return;
	int j = i*(n / (gridDim.x*blockDim.x))-1;
	for (int k = 1; k < (n / (gridDim.x*blockDim.x)); k++) {
		d_a[j + k] = d_a[j] + d_a[j + k];
	}
}


__host__
void compute_answers(double * a, double * b, long n) {
	double *d_a, *d_tmp;
	int thread = 1024, block = 2;

	//Allocate memory on GPU
	cudaMalloc(&d_a, n * sizeof(double));
	cudaMalloc(&d_tmp, n * sizeof(double)); //To hold temporary results

											//Copy content from CPU to GPU
	cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);

	//Copy content in to temporary array
	cudaMemcpy(d_tmp, d_a, n * sizeof(double), cudaMemcpyDeviceToDevice);

	//First pass
	local_sum << <block, thread >> > (d_a, n);
	cudaMemcpy(d_tmp, d_a, n * sizeof(double), cudaMemcpyDeviceToDevice);
	for (long p = 0; p <= log2l(2 * thread) - 1; p++) {
		add_kernel << <block, thread >> > (d_a, d_tmp, 1 << p, n);
		cudaMemcpy(d_a, d_tmp, n * sizeof(double), cudaMemcpyDeviceToDevice);
	}
	local_add << <block, thread >> > (d_a, n);

	//Second Pass
	local_sum << <block, thread >> > (d_a, n);
	cudaMemcpy(d_tmp, d_a, n * sizeof(double), cudaMemcpyDeviceToDevice);
	for (long p = 0; p <= log2l(2 * thread) - 1; p++) {
		add_kernel << <block, thread >> > (d_a, d_tmp, 1 << p, n);
		cudaMemcpy(d_a, d_tmp, n * sizeof(double), cudaMemcpyDeviceToDevice);
	}
	local_add << <block, thread >> > (d_a, n);

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
	return maxError / v[n - 1];
}

timespec time_diff(timespec start, timespec end)
{
    timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;
    }
    return temp;
}

int main() {
    printf("power seconds nanoseconds error\n");
    for (int power = 1; power < POWER; power++){

        double *a, *b;
        long n = 1 << power;

        
        //Allocate memory on CPU
        a = (double *)malloc(n * sizeof(double));
        b = (double *)malloc(n * sizeof(double));

        srand(clock());

        //Initialize values
        for (long i = 0; i < n; i++) {
            a[i] = ((double)(rand() % n)) / 100;
        }

        struct timespec start, end, difference;
        clock_gettime(CLOCK_MONOTONIC, &start);
        //Compute Answers
        compute_answers(a, b, n);
        clock_gettime(CLOCK_MONOTONIC, &end);

        difference = time_diff(start,end);
        printf("%d %d %ld %e\n",power+1, difference.tv_sec, difference.tv_nsec, verify_answers(a,b,n));

        //Verify Answers

        //Free memory on CPU
        free(a);
        free(b);
    }

	return 0;
}
