#include <cuda.h>
#include <stdio.h>
#include <time.h>  

#define Sizex 4
#define Sizey 4
//#define blockSize 16

void imprimirMatriz (int);

__global__ void MatrixMulKernel(int *d_M, int *d_N, int *d_P, int Width)
{	
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
		
	if((Row < Width) && (Col < Width))
	{
		int Pvalue = 0;
		for (int k = 0; k < Width; ++k)
		{
			Pvalue += d_M[Row * Width + k]*d_N[k * Width + Col];
		}	
		d_P[Row * Width + Col] = Pvalue;	
	}
}


void matrixMul(int *A, int *B, int *C, int n)
{
	int size = n*sizeof(int);
	int *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,size);
	cudaMalloc((void **)&d_B,size);
	cudaMalloc((void **)&d_C,size);
  	clock_t t3;
  	t3 = clock();
	//Copio los datos al dispositivo
	cudaMemcpy( d_A, A, size, cudaMemcpyHostToDevice);	
	cudaMemcpy( d_B, B, size, cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	dim3 dimBlock = (Sizex, Sizey);
	dim3 dimGrid = (ceil((float)n/(float)dimBlock.x), ceil((float)n/(float)dimBlock.y));
	
	MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, size);	
	
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
  		printf("\n");
  	}
  	t3 = clock() - t3;
  	printf ("\nTiempo GPU Multiplication: (%f seconds).\n",((float)t3)/CLOCKS_PER_SEC);

	//libera memoria del dispositivo
	cudaFree(d_A);																			
	cudaFree(d_B);
	cudaFree(d_C);
}

void agregarDatos(int *A, int *B)
{
	int i;
	for(i = 0; i < Sizex*Sizey; i++)
	{
		//A[i] = rand() % 5;
		//B[i] = rand() % 9;
		A[i] = 1;
		B[i] = 1;
	}
	printf ("Datos Agregados...\n");
}

void imprimirMatriz (int *A)
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
	
	printf("\n...Matriz A...\n");
	imprimirMatriz (A);
	
	printf("\n...Matriz B...\n");
	imprimirMatriz (B);
	
	printf("\n...Multiplicacion de matrices...\n");
	matrixMul(A, B, C, sizetotal);
	
	return 0;
}