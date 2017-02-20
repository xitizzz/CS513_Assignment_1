#include <stdio.h>
#include <stdlib.h>
#include <time.h>
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
void compute_answers(double * a, double * b, long n)
{
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
double verify_answers(double *a, double * b, long n)
{
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

void sum(double* a, double* b, const int n) {
    //Given an array a[0...n-1], you need to compute b[0...n-1],
    //where b[i] = (i+1)*a[0] + i*a[1] + ... + 2*a[i-1] + a[i]
    //Note that b is NOT initialized with 0, be careful!
    //Write your CUDA code starting from here
    //Add any functions (e.g., device function) you want within this file

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

    return;
}

int main(int argc, const char * argv[]) {

    if (argc != 2) {
        printf("The argument is wrong! Execute your program with only input file name!\n");
        return 1;
    }
    
    //Dummy code for creating a random input vectors
    //Convenient for the text purpose
    //Please comment out when you submit your code!!!!!!!!! 	
    /*FILE *fp = fopen(argv[1], "w");
    if (fp == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    int n = 1 << 24;
    fprintf(fp, "%d\n", n);
    srand(time(NULL));
    for (int i=0; i<n; i++)
        fprintf(fp, "%lg\n", ((double)(rand() % n))/100);
    fclose(fp);
    printf("Finished writing\n");*/
    
    //Read input from input file specified by user
    FILE* fp = fopen(argv[1], "r");
    if (fp == NULL) {
        printf("The file can not be opened or does not exist!\n");
        return 1;
    }
    int n;
    fscanf(fp, "%d\n", &n);
    printf("%d\n", n);
    double* a = (double *)malloc(n*sizeof(double));
    double* b = (double *)malloc(n*sizeof(double));
    for (int i=0; i<n; i++) {
        fscanf(fp, "%lg\n", &a[i]);
    }
    fclose(fp);
    
    //Main function
    sum(a, b, n);
    
    //Write b into output file
    fp = fopen("output.txt","w");
    if (fp == NULL) {
        printf("The file can not be created!\n");
        return 1;
    }
    fprintf(fp, "%d\n", n);
    for (int i=0; i<n; i++)
        fprintf(fp, "%lg\n", b[i]);
    fclose(fp);
    free(a);
    free(b);
    printf("Done...\n");
    return 0;
}
