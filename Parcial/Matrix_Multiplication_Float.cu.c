#include <cuda.h>
#include <stdio.h>
#include <time.h>  

#define blockSize 32
#define TILE_WIDTH 32


// Paralela
// Multiplicacion Kernel sin Tiles
__global__ void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int m, int n, int p)
{	
	int Row = blockIdx.y * blockDim.y + threadIdx.y; // Index de la fila resultante
	int Col = blockIdx.x * blockDim.x + threadIdx.x; // Index de la columna resultante
		
	if((Row < m) && (Col < p))
	{
		float Pvalue = 0;
		for (int k = 0; k < n; k++)
		{
			Pvalue += d_M[Row * n + k]*d_N[k * p + Col];
		}	
		d_P[Row * p + Col] = Pvalue;	
	}
}



// Funcion que llama Multiplicacion Kernel sin Tiles
void matrixMul(float *A, float *B, float *C, int m, int n, int p)
{
	float *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,m*n*sizeof(float));
	cudaMalloc((void **)&d_B,n*p*sizeof(float));
	cudaMalloc((void **)&d_C,m*p*sizeof(float));
  	clock_t t3;
  	t3 = clock();
	//Copio los datos al dispositivo
	cudaMemcpy( d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy( d_B, B, n*p*sizeof(float), cudaMemcpyHostToDevice);
	// Ejecuto el Kernel 
	dim3 dimBlock(blockSize, blockSize, 1);
	// p => quiere decir cuantas columnas
	// m => quiere decir cuantas filas
	dim3 dimGrid(ceil((float)p/dimBlock.x), ceil((float)m/dimBlock.y),1); // Aca se le envia la cantidad total de memoria

	MatrixMulKernel<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, p);	
	
	cudaMemcpy( C, d_C, m*p*sizeof(float), cudaMemcpyDeviceToHost);
 
  	t3 = clock() - t3;
  	printf ("\nMULTIPLICATION WITHOUT TILING: (%f seconds).\n",((float)t3)/CLOCKS_PER_SEC);

	//libera memoria del dispositivo
	cudaFree(d_A);																			
	cudaFree(d_B);
	cudaFree(d_C);
}


// Paralela
// Multiplicacion Kernel con Tiles 
__global__ void MatrixMulKernelTiles(float *d_MatA, float *d_MatB, float *d_MatR, int m, int n, int o)
{	
	// ds => data shared
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	
	int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;

	for (int k = 0; k < (n + TILE_WIDTH - 1)/(TILE_WIDTH); ++k)
	{
		// Cargamos la Matriz A y la matriz B a Memoria compartida
		// Verificamos q no se salga del limite
		// ty Menor al numero de columnas y Row menor al numero de filas
		if (k * TILE_WIDTH + tx < n && Row < m)
		{
			Mds[ty][tx] = d_MatA[Row * n + k * TILE_WIDTH + tx];
		}
		else
		{
      		Mds[ty][tx] = 0;
      	}
      	if (k * TILE_WIDTH + ty < n && Col < o)
      	{
        	Nds[ty][tx] = d_MatB[(k * TILE_WIDTH + ty) * o + Col];
      	}
      	else
      	{
	        Nds[ty][tx] =0;
      	}
      	__syncthreads(); // primer punto de sincronizacion 
      	for(int k = 0; k < TILE_WIDTH; ++k)
      	{
	        Pvalue += Mds[ty][k] * Nds[k][tx];
      	}
      	__syncthreads(); // Segundo punto de sincronizacion 
	}
	if((Row < m) && (Col < o))
	{
		d_MatR[Row * o + Col] = Pvalue;
	}
}



// Funcion que llama Multiplicacion Kernel sin Tiles
void matrixMulTiles(float *A, float *B, float *C, int m, int n, int p)
{
	float *d_A, *d_B, *d_C;
	//Reservo Memoria en el dispositivo
	cudaMalloc((void **)&d_A,m*n*sizeof(float));
	cudaMalloc((void **)&d_B,n*p*sizeof(float));
	cudaMalloc((void **)&d_C,m*p*sizeof(float));
  	clock_t t4;
  	t4 = clock();
	//Copio los datos al dispositivo
	cudaMemcpy( d_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice);	
	cudaMemcpy( d_B, B, n*p*sizeof(float), cudaMemcpyHostToDevice);
	// Ejecuto el Kernel (del dispositivo)
	dim3 dimBlock(blockSize, blockSize,1);
	dim3 dimGrid(ceil((float)p/dimBlock.x), ceil((float)m/dimBlock.y),1);
	
	MatrixMulKernelTiles<<< dimGrid, dimBlock >>>(d_A, d_B, d_C, m, n, p);	
	
	cudaMemcpy( C, d_C, m*p*sizeof(float), cudaMemcpyDeviceToHost);
  
  	t4 = clock() - t4;
  	printf ("\nMULTIPLICATION WITH TILING: (%f seconds).\n",((float)t4)/CLOCKS_PER_SEC);
	//libera memoria del dispositivo
	cudaFree(d_A);																			
	cudaFree(d_B);
	cudaFree(d_C);
}

// Multiplicacion de matrices secuencial
//void multMatrixsequential (int *h_MatA, int *h_MatB, int *h_MatR, int n, int m, int o)
void multMatrixsequential (float *h_MatA, float *h_MatB, float *h_MatR, int m, int n, int o)
{
	clock_t t5;
	t5 = clock();
	for (int i = 0; i < m; i++) // Mueve entre fila de la matriz inicial
	{
		for (int j = 0; j < o; j++) // Se mueve entre columnas
		{
			float sum = 0;
			for (int k = 0; k < n; k++)
			{
				sum += h_MatA[n * i + k] * h_MatB[o * k + j];
			}
			h_MatR[o * i + j] = sum;
		}
	}
  	t5 = clock() - t5;
	printf ("\nMULTIPLICACION SECUENCIAL: (%f seconds).\n",((float)t5)/CLOCKS_PER_SEC);
}

// Funcion que agrega los datos a las matrices
void agregarDatos(float *A, int m, int n)
{
	for(int i = 0; i < m*n; i++)
	{
		A[i] = rand() % 3;
		//A[i] = 1;
	}
}

// IMPRIMIR MATRIZ
void imprimirMatriz (float *A, int a, int b)
{
    for (int i = 0; i < b; i++)
    {
		for (int j = 0; j < a; j++)
		{
			if (A[i*b + j] < 10)
			{
				printf ("%f  |  ", A[i*a + j]);
			}
			else
			{
				printf ("%f | ", A[i*a + j]);	
			}
		}
    printf("\n");
    }
}

int compararMatrices(float *MA, float *MB, int m, int n)
{
	int tam = m*n;
	for (int i = 0; i < tam; ++i)
	{
		if (MA[i] != MB[i])
		{
			printf("\n...EL resultado de la matriz Secuencial y Paralela NO son iguales...\n");
			return 0;
			//break;
		}
	}
	printf("\n...Son iguales las matrices comparadas...\n");
	return 0;
}

int main()
{
	//Definicion tamaño de  matrices
	// mxn X nxp = mxp
	int m = 12;
	int n = 16;
	int p = 32;
	float *A, *B, *RS, *RP, *RPT;
	A = (float*)malloc(m*n*sizeof(float));	
	B = (float*)malloc(n*p*sizeof(float)); 	
	RS = (float*)malloc(m*p*sizeof(float)); 
	RP = (float*)malloc(m*p*sizeof(float));
	RPT = (float*)malloc(m*p*sizeof(float));
	agregarDatos(A, m, n);
	agregarDatos(B, p, n);	
	multMatrixsequential(A, B, RS, m, n, p);
	matrixMul(A, B, RP, m, n, p);
	matrixMulTiles(A, B, RPT, m, n, p);
	printf("\nComparacion de Matriz secuencial con Paralela");
	compararMatrices(RS, RP, m, p);
	printf("\nComparacion de Matriz secuencial con Tiles");
	compararMatrices(RS, RPT, m, p);
	free (A);
	free (B);
	free (RS);
	free (RP);
	free (RPT);
	return 0;
}