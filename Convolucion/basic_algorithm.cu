#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <math.h> 

#define BLOCK_SIZE  32
#define TILE_SIZE  32
#define MAX_MASK_WIDTH  10

__constant__ float M[MAX_MASK_WIDTH];

/************** Imprimir Vector *******************/
void imprimiVector (float *A, int length)
{
	for (int i = 0; i < length; i++)
	{
		printf ("%f |", A[i]);
	}
	printf("\n");
}

/************* Funcion que llena el vector ************************/
 void agregarDatos (float *Matrix, int length)
{
	for(int i = 0; i < length; i++)
	{
		Matrix[i] = rand() % 10;
	}
}

/****************** Comparar los vectores ************************/
int compararVector(float *M_A, float *M_B, int length)
{
	for (int i = 0; i < length; ++i)
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

void basicConvolution (float *input, float *output, float *mask, int mask_length, int length)
{
	int inicio = 0;
	float temp = 0.0;
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < mask_length; j++)
		{
			inicio = i - (mask_length / 2);
			if (inicio + j >= 0 && inicio + j < length)
			{
				temp += input[inicio + j] * mask[j];
			}
		}
		output[i] = temp;
		temp = 0.0;
	}
}


/***************** Convolucion Kernel Basico *******************/
__global__ void convolutionBasicKernel(float *N, float *M, float *P, int Mask_Width, int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
    float Pvalue = 0;
    int N_start_point = i - (Mask_Width/2);
    for (int j = 0; j < Mask_Width; j++)
    {
    	if (N_start_point + j >= 0 && N_start_point + j < Width)
      	{
        	Pvalue += N[N_start_point + j]*M[j];
      	}
    }
    P[i] = Pvalue;
}

/********** Convolucion Memoria constante y Caching *****************/
__global__ void convolutionKernelConstant(float *N, float *P, int Mask_Width, int Width)
 {
   int i = blockIdx.x*blockDim.x + threadIdx.x;
   float Pvalue = 0;
   int N_start_point = i - (Mask_Width/2);
   for (int j = 0; j < Mask_Width; j++)
   {
     if (N_start_point + j >= 0 && N_start_point + j < Width)
     {
       Pvalue += N[N_start_point + j]*M[j];
     }
   }
   P[i] = Pvalue;
}

/********* Convolucion con "Halo Elements" ********************/
__global__ void convolutionKernelShared(float *N, float *P, int Mask_Width, int Width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	__shared__ float N_ds[TILE_SIZE + MAX_MASK_WIDTH - 1];

	int n = Mask_Width/2;

	int halo_index_left = (blockIdx.x - 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x >= blockDim.x - n)
	{
		N_ds[threadIdx.x - (blockDim.x - n)] = (halo_index_left < 0) ? 0 : N[halo_index_left];
	}
	N_ds[n + threadIdx.x] = N[blockIdx.x*blockDim.x + threadIdx.x];
	
	int halo_index_right = (blockIdx.x + 1)*blockDim.x + threadIdx.x;
	if (threadIdx.x < n)
	{
		N_ds[n + blockDim.x + threadIdx.x] = (halo_index_right >= Width) ? 0 : N[halo_index_right];
	}

	__syncthreads();

	float Pvalue = 0;
	for(int j = 0; j < Mask_Width; j++)
	{
		Pvalue += N_ds[threadIdx.x + j]*M[j];
	}
	P[i] = Pvalue;
}

/*****************************/
/*** LLAMADOS A LOS KERNEL ***/
/*****************************/

// Llamada al convolutionBasicKernel
void convolutionCall (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_mask;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_mask, mask_length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpy (d_mask, mask, mask_length * sizeof (float), cudaMemcpyHostToDevice);

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionBasicKernel<<<dimGrid, dimBlock>>> (d_input, d_mask, d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_mask);
  cudaFree (d_output);
}

// llamada al convolutionKernelConstant
void convolutionCallConstant (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (M, mask, mask_length * sizeof (float));

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionKernelConstant<<<dimGrid, dimBlock>>> (d_input,d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_output);
}

// llamada al convolutionKernelShared
void convolutionCallWithTilesComplex (float *input, float *output, float *mask, int mask_length, int length)
{
  float *d_input;
  float *d_output;
  float block_size = BLOCK_SIZE;//The compiler doesn't let me cast the variable

  cudaMalloc(&d_input, length * sizeof(float));
  cudaMalloc(&d_output, length * sizeof(float));

  cudaMemcpy (d_input, input, length * sizeof (float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol (M, mask, mask_length * sizeof (float));

  dim3 dimGrid (ceil (length / block_size), 1, 1);
  dim3 dimBlock (block_size, 1, 1);

  convolutionKernelShared<<<dimGrid, dimBlock>>> (d_input,d_output, mask_length, length);
  cudaDeviceSynchronize();

  cudaMemcpy (output, d_output, length * sizeof (float), cudaMemcpyDeviceToHost);
  cudaFree (d_input);
  cudaFree (d_output);
}


// =======================================================//

int main()
{
	// Se tomaran 30 tamaños diferentes para el vector
	for (int i = 0; i < 30; i++)
	{
		printf("\nEjecucion numero %d\n", (i+1));
		// Tamaño del vector
		unsigned int length = pow (2, i);
		printf("Tamaño del vector: %d\n", length);
		// Tamaño de la mascara
		int mask_length = 5;
		clock_t inicio, fin;
		double basico, kernelBasic, kernelConstant, kernelShared;//, optimizacion;

		float *A = (float *) malloc(length * sizeof(float));
	    float *mask = (float *) malloc(mask_length * sizeof(float));
	    float *Cserial = (float *) malloc(length * sizeof(float));
	    float *Cparallel = (float *) malloc(length * sizeof(float));
	    float *CparallelConstant = (float *) malloc (length * sizeof(float));
	    float *CparallelWithTiles = (float *) malloc(length * sizeof(float));

	    // Llenar el vector general y la mascara
	    agregarDatos(A, length);
	    agregarDatos(mask, mask_length);

	    // ------------------------------------------

	    //printf("Resultado Secuencial\n");
	    inicio = clock();
	    basicConvolution(A, Cserial, mask, mask_length, length);
	    fin = clock();
	    basico = (((double) (fin - inicio)) / CLOCKS_PER_SEC );
	    printf("El proceso Secuencial tomo: %f segundos\n", basico);

	    //--------------------------------------------
	    //printf("Resultado Kernel Basico\n");
	    inicio = clock();
	    convolutionCall(A, Cparallel, mask, mask_length, length);
	    fin = clock();
	    kernelBasic = (((double) (fin - inicio)) / CLOCKS_PER_SEC );
	    printf("El proceso paralelo basico tomo: %f segundos\n", kernelBasic);
	    //optimizacion = basico / kernelBasic;
	    //printf("La aceleracion es: %f\n", optimizacion);
	    compararVector (Cserial, Cparallel, length);

		//--------------------------------------------
	    //printf("Resultado Kernel con variable constante y Caching\n");
	    inicio = clock();
	    convolutionCallConstant(A, CparallelConstant, mask, mask_length, length);
	    fin = clock();
	    kernelConstant = (((double) (fin - inicio)) / CLOCKS_PER_SEC );
	    printf("El proceso con variable constante y Caching tomo: %f segundos\n", kernelConstant);
	    //optimizacion = basico / kernelConstant;
	    //printf("La aceleracion es: %f\n", optimizacion);
	    compararVector (Cserial, CparallelConstant, length);	    

	    //--------------------------------------------
	    //printf("Resultado Kernel Tiles 1D con Halo Elements\n");
	    inicio = clock();
	    convolutionCallWithTilesComplex(A, CparallelWithTiles, mask, mask_length, length);
	    fin = clock();
	    kernelShared = (((double) (fin - inicio)) / CLOCKS_PER_SEC );
	    printf("El proceso con Tiles tomo: %f segundos\n", kernelShared);
	    //optimizacion = basico / kernelShared;
	    //printf("La aceleracion es: %f\n", optimizacion);
	    compararVector (Cserial, CparallelWithTiles, length);

		free(A);
		free(mask);
		free(Cserial);
		free(Cparallel);
		free(CparallelConstant);
		free(CparallelWithTiles);
	}
}