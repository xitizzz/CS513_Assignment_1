#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include <time.h> //for clock_gettime, high precision timer
#include <stdio.h> //for printf

#define POWER 28
#define THREAD 1024

using namespace std;
struct timespec start[2*POWER], end[2*POWER], difference[2*POWER]; //array of timespecs, one per iteration per pass

__global__
void add_kernel(double * d_a, double * d_tmp, long k, long n) {
	long i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i + k < n) {
		d_tmp[i + k] = d_a[i + k] + d_a[i];
	}
}

__host__
void compute_answers(double * a, double * b, long n, int power) {
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
	for (long p = 0; p <= power; p++) {
        clock_gettime(CLOCK_MONOTONIC, &start[p]);
		add_kernel << <(n + THREAD - 1) / THREAD, THREAD >> > (d_a, d_tmp, 1 << p, n);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end[p]);
		cudaMemcpy(d_a, d_tmp, n * sizeof(double), cudaMemcpyDeviceToDevice);
	}

	cudaMemcpy(d_tmp, d_a, n * sizeof(double), cudaMemcpyDeviceToDevice);

	//Second pass
	//Launch kernel log n times 
	for (long p = 0; p <= power; p++) {
        clock_gettime(CLOCK_MONOTONIC, &start[p+POWER]);
		add_kernel << <(n + THREAD - 1) / THREAD, THREAD >> > (d_a, d_tmp, 1 << p, n);
        cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end[p+POWER]);
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

        //Initialize values
        for (long i = 0; i < n; i++) {
            a[i] = ((double)(rand() % n)) / 100;
        }

        //Initialize timers
        for (int i = 0; i < 2*POWER; i++){
            start[i].tv_sec = 0;
            start[i].tv_nsec = 0;
            end[i].tv_sec = 0;
            end[i].tv_nsec = 0;
            difference[i].tv_sec = 0;
            difference[i].tv_nsec = 0;
        }


        //Compute Answers
        compute_answers(a, b, n, power);

        cudaDeviceSynchronize();
        time_t diff_sec = 0;
        long diff_nsec = 0;

        for (int i = 0; i < 2*POWER; i++){
            difference[i] = time_diff(start[i],end[i]);
            diff_sec += difference[i].tv_sec;
            diff_nsec += difference[i].tv_nsec;
        }

        printf("%d %d %ld %e\n",power+1, diff_sec, diff_nsec, verify_answers(a,b,n));

        //Free memory on CPU
        free(a);
        free(b);
    }
	return 0;
}
