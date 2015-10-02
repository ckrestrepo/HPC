#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 1024
#define blockSize 1024


__global__ void vecAdd(int *A, int *B, int *C)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < TAM)
	{
		C[i] = A[i] + B[i];
	}
}

int vectorAdd(int *A, int *B, int *C)
{
	int size = TAM*sizeof(int);
	int *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
  
	clock_t t2;
  	t2 = clock();

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  
  	float dimGrid = ceil((float)TAM / (float)blockSize);

	vecAdd<<<dimGrid, TAM>>>(d_A, d_B, d_C);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

  	t2 = clock() - t2;
  	printf ("Tiempo GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	return 0;
}

void sumarSecuencial(int *A, int *B, int *C)
{
    clock_t t;
    t = clock();
    for(int i = 0; i < TAM; i++)
    {
	    C[i]= A[i] + B[i];
    }
   	t = clock() - t;
   	printf ("Tiempo CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
 }

 void agregarDatos (int *Matrix)
{
	for(int i = 0; i < TAM; i++)
	{
		Matrix[i] = rand() % 5;
	}
}

int compararVector(int *M_A, int *M_B)
{
	for (int i = 0; i < TAM; ++i)
	{
		if (M_A[i] != M_B[i])
		{
			printf("Los Vectores son diferentes\n");
			return 0;
		}
	}
	printf("\nLos Vectores son iguales\n");
	return 0;
}
    

int main()
{
	int * A;
	int * B;
	int * CS;
	int * CP;

	A = (int*)malloc(TAM*sizeof(int));
	B = (int*)malloc(TAM*sizeof(int));
	CS = (int*)malloc(TAM*sizeof(int));
	CP = (int*)malloc(TAM*sizeof(int));

	agregarDatos(A);
	agregarDatos(B);

	printf("\nSuma de vector Paralela\n");
  	vectorAdd(A,B,CS);
  	printf("\nSuma de vector Secuencial\n");
  	sumarSecuencial(A,B,CP);
  	compararVector(CS, CP);

	return 0;
}
