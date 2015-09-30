#include <cuda.h>
#include <stdio.h>
#include <time.h>  

#define Sizex 25
#define Sizey 25
#define blockSize 1024

void imprimirMatriz (int, int);

__global__ void MatrixSumKernel(float *d_M, float *d_N, float *d_P, int Width)
{	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	int index = Col + Row * Width;
	
	if((Row < Width) && (Col < Width))
	{
		d_P[index] = d_M[index] + d_N[index];
	}
}

void matrixAdd(int *A, int *B, int *C, int n)
{
	int size = n*sizeof(int);
	float *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,size);
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);
  	clock_t t2;
  	t2 = clock();
	//Copio los datos al dispositivo
	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);	
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	//dim3 dimBlock = 1024;
	dim3 dimGrid = ((int)ceil((float)blockSize/(float)32), (int)ceil((float)blockSize/(float)32), 1);
	
	MatrixSumKernel<<< dimGrid, blockSize >>>(d_A, d_B, d_C, size);	
  
	cudaMemcpy( C, d_C, size, cudaMemcpyDeviceToHost);
	for(int i = 0; i < Sizex; i++ )
  	{
	  	for (int j = 0; j < Sizey; j++)
	  	{
	  		if (C[i*Sizey + j] < 10)
			{
				printf ("%d  |  ", C[i*Sizey + j]);
			}
			else
			{
				printf ("%d | ", C[i*Sizey + j]);	
			}
  		}
  		printf(" - Ciclo [%d] \n", (i+1));
  	}
  	t2 = clock() - t2;
  	printf ("\nTiempo GPU: (%f seconds).\n",((float)t2)/CLOCKS_PER_SEC);

	//libera memoria del dispositivo
	cudaFree(d_A);																			
	cudaFree(d_B);
	cudaFree(d_C);
}

void sumarMatriz(int *A, int *B, int *C)
{
	clock_t t1;
   	t1 = clock();
	int i, j;
  	for(i = 0; i < Sizex; i++ )
  	{
	  	for (j = 0; j < Sizey; j++)
	  	{
	  		C[i*Sizey + j] = A[i*Sizey + j] + B[i*Sizey + j];
	  		if (C[i*Sizey + j] < 10)
			{
				printf ("%d  |  ", C[i*Sizey + j]);
			}
			else
			{
				printf ("%d | ", C[i*Sizey + j]);	
			}
  		}
  		printf("\n");
  	}
  	t1 = clock() - t1;
  	printf ("Tiempo CPU: (%f seconds).\n",((float)t1)/CLOCKS_PER_SEC);
}

void agregarDatos(int *A, int *B)
{
	int i;
	for(i = 0; i < Sizex*Sizey; i++)
	{
		A[i] = rand() % 5;
		B[i] = rand() % 9;
		//A[i] = i;
		//B[i] = i;
	}
	printf ("Datos Agregados...\n");
}

void imprimirMatriz (int *A, int n)
{
    for (int i = 0; i < Sizex; i++)
    {
		for (int j = 0; j < Sizey; j++)
		{
			if (A[i*Sizex + j] < 10)
			{
				printf ("%d  |  ", A[i*Sizex + j]);
			}
			else
			{
				printf ("%d | ", A[i*Sizex + j]);	
			}
			
		}
    printf("\n");
    }
}

int main()
{
	int *A, *B, *C;
	int sizetotal = Sizex * Sizey;
	A = (int*)malloc(sizetotal*sizeof(int));
	B = (int*)malloc(sizetotal*sizeof(int));
	C = (int*)malloc(sizetotal*sizeof(int));
	
	agregarDatos(A, B);
	
	printf("...Matriz A...\n");
	imprimirMatriz (A, sizetotal);
	
	printf("...Matriz B...\n");
	imprimirMatriz (B, sizetotal);
	
	printf("\n...Suma de matrices Paralela...\n");
	matrixAdd(A,B,C, sizetotal);
	
	printf("\n...Suma de matrices Secuencial...\n");
	sumarMatriz(A, B, C);
	
	return 0;
}