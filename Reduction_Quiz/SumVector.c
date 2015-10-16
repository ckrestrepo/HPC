#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cstdlib>
#include <time.h>
#include <math.h> 

#define BLOCK_SIZE 1024 

/*********** Sumatoria Vector Secuencial ****************/
double SumVecSerial (double *A, int length)
{
  double sum = 0;
  for (int i = 0; i < length; i++)
  {
    sum = sum + A[i];
  }
  return sum;
}

/************** Imprimir Vector *******************/
void printVector (double *A, int length)
{
  for (int i = 0; i < length; i++)
  {
	  printf ("%f |", A[i]);
  }
  printf("\n");
}

/************* Funcion que llena el vector ************************/
void fillVector (double *A, int length)
{
  for (int i=0; i<length; i++)
  {
	  A[i] = rand() % 10;
  }
}

/************** Funcion que compara el resultado  *******************/
void resultCompare(double A, double  *B)
{
  if(fabs(A-B[0]) < 0.1)// taking in count decimal precision
  {
	  printf("Bien...\n");
  } 
  else
  {
	  printf("No tan bien...\n");
  }
}

/******************** REDUCTION KERNEL ******************/
//Paralelo
__global__ void reduceKernel(double *g_idata, double *g_odata, int length)
{
  __shared__ double sdata[BLOCK_SIZE];
  // Cada hilo carga un elemento de memoria global a memoria compartida
  unsigned int tid = threadIdx.x; //ID del hilo
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i < length)
  {
    sdata[tid] = g_idata[i];
  } 
  else
  {
    sdata[tid] = 0.0;
  }
  __syncthreads();
  // Hacer reduccion en memoria compartida
  for (unsigned int s = blockDim.x/2; s > 0; s>>=1)
  {
    if (tid < s) 
	{
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // Escribir el resultado para este bloque a Memoria Global
  if (tid == 0)
  {
    g_odata[blockIdx.x] = sdata[0];

  }
}

void vectorItemsAdd(double *A, double *B, int length)
{
  double * d_A;//Device variables
  double * d_B;

  cudaMalloc(&d_A,length*sizeof(double));
  cudaMalloc(&d_B,length*sizeof(double));

  cudaMemcpy(d_A, A,length*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B,length*sizeof(double),cudaMemcpyHostToDevice);

  int aux = length;

  while(aux > 1)
  {
     dim3 dimBlock(BLOCK_SIZE,1,1);
     int grid = ceil(aux/(double)BLOCK_SIZE); //Cast needed to make this work
     dim3 dimGrid(grid,1,1);
     reduceKernel<<<dimGrid,dimBlock>>>(d_A,d_B,aux);
     cudaDeviceSynchronize();
	 
     cudaMemcpy(d_A,d_B,length*sizeof(double),cudaMemcpyDeviceToDevice);
     aux=ceil(aux/(double)BLOCK_SIZE);
  }

  cudaMemcpy(B,d_B,length*sizeof(double),cudaMemcpyDeviceToHost);

  cudaFree(d_A);
  cudaFree(d_B);
}


int main ()
{

 for(int i = 0; i < 15; i++)
 {
	printf("Ejecucion numero: %d\n", i);
  	unsigned int l = pow(2,i); //Vector's length, variable in every execution to get the test values faster
	printf("Tamaño del vector: %d\n", l);
	clock_t start, finish; //Clock variables
	double elapsedSecuential, elapsedParallel, optimization;

   	double *A = (double *) malloc(l * sizeof(double));
   	double *B = (double *) malloc(l * sizeof(double));

   fillVector(A,l);
   fillVector(B,l);

   //========================= SERIAL ==========================================
   start = clock();
   double sum = SumVecSerial(A,l);
   finish = clock();
   printf("El resultado es: %f\n", sum);
   elapsedSecuential = (((double) (finish - start)) / CLOCKS_PER_SEC );
   printf("El proceso secuencial tomo: %f segundos en ejecutar\n\n", elapsedSecuential);

   //======================= PARALLEL ==========================================
   start = clock();
   vectorItemsAdd(A,B,l);
   finish = clock();
   printf("El resultado es: %f\n", B[0]);
   elapsedParallel = (((double) (finish - start)) / CLOCKS_PER_SEC );
   printf("El proceso paralelo tomo: %f segundos en ejecutar\n\n", elapsedParallel);

   optimization = elapsedSecuential/elapsedParallel;
   printf("Aceleracion obtenida: %f\n", optimization);
   resultCompare(sum, B);
   free(A);
   free(B);
 }
}