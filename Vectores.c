#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#define TAM 30
#define blockSize 1024


__global__ void vecAdd(int *A, int *B, int *C, int n)
{
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < n)
	{
		C[i] = A[i] + B[i];
	}
}

int vectorAdd(int *A, int *B, int *C, int n)
{
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	cudaMalloc((void **)&d_A, size);
	cudaMalloc((void **)&d_B, size);
	cudaMalloc((void **)&d_C, size);
  
	clock_t t2;
  	t2 = clock();

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  
  	float dimGrid = ceil((float)TAM / (float)blockSize);

	vecAdd<<<dimGrid, n>>>(d_A, d_B, d_C, n);
	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < n; i++)
    {
	    C[i] = A[i] + B[i];
	    printf (" %d | ", C[i]);
    }  

  	t2 = clock() - t2;
  	printf ("\nTiempo GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);
	
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return 0;
}

void sumar(int *A, int *B, int *C, int n)
{
   
    clock_t t;
    t = clock();
    for(int i = 0; i < n; i++)
    {
	    C[i]= A[i] + B[i];
	    printf (" %d | ", C[i]);
    }
   	t = clock() - t;
   	printf ("\nTiempo CPU: (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);
 }
    

int main()
{
	int n; //longitud del vector
	int * A;
	int * B;
	int * C;
  	n = TAM;

	A = (int*)malloc( n*sizeof(int) );
	B = (int*)malloc( n*sizeof(int) );
	C = (int*)malloc( n*sizeof(int) );
	printf("...Vector A...\n");
	for(int i = 0; i < n; i++)
	{
		A[i] = rand() % 10 ;
    	printf("%d | ",A[i]);
	}
	printf("\n...Vector B...\n");
	for (int i = 0; i < n; ++i)
	{
		B[i] = rand() % 10;
		printf("%d | ",B[i]);
	}
		
	//vecAddGPU(A,B,C);
	printf("\nSuma de vector Paralela\n");
  	vectorAdd(A,B,C,n);
  	printf("\nSuma de vector Secuencial\n");
  	sumar(A,B,C,n);
	return 0;
}