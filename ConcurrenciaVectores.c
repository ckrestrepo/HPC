#include "stdio.h"
#include "cuda.h"
#include "stdlib.h"
#include "time.h"
#include "cstdlib"
#include "math.h"
#define SIZE 2000
#define BLOCKSIZE 1024


/******************* NORMAL KERNEL ******************/
/****************** SUMA DE VECTORES ****************/
__global__ void vecAdd(int *A, int *B, int *C, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
	{
		C[i] = A[i] + B[i];
  	}
}

int vectorAddGPU( int *A, int *B, int *C, int n)
{
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al dispositivo
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	float dimGrid = ceil((float)SIZE / (float)BLOCKSIZE);

	vecAdd<<< dimGrid, BLOCKSIZE >>>(d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

/******************** END SESION ***********************/

__global__ void scan(float *g_odata, float *g_idata, int n)
{
	__shared__ float temp[32]; // allocated on invocation
	int thid = threadIdx.x;
	int pout = 0, pin = 1;
	// Load input into shared memory.
	 // This is exclusive scan, so shift right by one
	 // and set first element to 0
	temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
	__syncthreads();
	for (int offset = 1; offset < n; offset *= 2)
	{
	  pout = 1 - pout; // swap double buffer indices
	  pin = 1 - pout;
	  if (thid >= offset)
	    temp[pout*n+thid] += temp[pin*n+thid - offset];
	  else
	    temp[pout*n+thid] = temp[pin*n+thid];
	  __syncthreads();
	}
	g_odata[thid] = temp[pout*n+thid]; // write output
}

__global__ void sumaCurrency(int *g_idata, int *g_odata)
{
	extern __shared__ int sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s=1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
		// write result for this block to global mem
	if
	(tid == 0) g_odata[blockIdx.x] = sdata[0];
}


int vectorAddGPUCurrency( int *A, int *B, int *C, int n){
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
	//Copio los datos al dispositivo
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	float dimGrid = ceil((float)SIZE / (float)BLOCKSIZE);
	
	sumaCurrency<<< 1, BLOCKSIZE >>>(d_A, d_C);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

int vectorAddCPU( int *A, int *B, int *C, int n)
{
	for(int i = 0; i < n; i++)
	{
		C[i] = A[i] + B[i];
	}
	return 0;
}

int main()
{
	int SIZES[] = {512, 1024, 3000, 5000, 1000000, 5000000};
	for (int j = 0; j < sizeof(SIZES)/sizeof(SIZES[0]); ++j)
	{
		int *A=(int *) malloc(SIZES[j]*sizeof(int));
		int *B=(int *) malloc(SIZES[j]*sizeof(int));
		int *CS=(int *) malloc(SIZES[j]*sizeof(int));	// Secuencial
		int *CP=(int *) malloc(SIZES[j]*sizeof(int));	// Paralelo
		int *CC=(int *) malloc(SIZES[j]*sizeof(int));	// Concurrente
		clock_t inicioCPU, inicioGPU, inicioGPUC, finCPU, finGPU, finGPUC;
		int i;
		for(i = 0; i < SIZES[j]; i++)
		{
			A[i]=rand()%9;
			B[i]=rand()%5;
		}
		// Ejecuto por GPU
		inicioGPU=clock();
		vectorAddGPU(A, B, CP, SIZES[j]);
		finGPU = clock();
		// Ejecuto por CPU
		inicioCPU=clock();
		vectorAddCPU(A, B, CS, SIZES[j]);
		finCPU=clock();

		// Ejecuto por GPU Concurrente
		inicioGPUC=clock();
		vectorAddGPUCurrency(A, B, CC, SIZES[j]);
		finGPUC=clock();

		printf("Size %d\n", SIZES[j]);

		printf("El tiempo GPU es: %f\n",(double)(finGPU - inicioGPU) / CLOCKS_PER_SEC);
		printf("El tiempo CPU es: %f\n",(double)(finCPU - inicioCPU) / CLOCKS_PER_SEC);
		printf("El tiempo GPU Concurrente es: %f\n",(double)(finGPUC - inicioGPUC) / CLOCKS_PER_SEC);
		free(A);
		free(B);
		free(CS);
		free(CP);
		free(CC);
	}
	return 0;
}
