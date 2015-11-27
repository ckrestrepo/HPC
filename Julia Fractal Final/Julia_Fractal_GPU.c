#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

#define DIM 400
#define BLOCK_SIZE 32
using namespace std;
using namespace cv;

// Funciones para el Device
// el __device__ indica que el codigo correra en GPU y no en el Host

struct cuComplex 
{
	float r;
	float i;
	__device__ cuComplex( float a, float b ) : r(a), i(b) 
	{

	}

	__device__ float magnitude2( void ) 
	{ 
		return r * r + i * i; 
	}

	__device__ cuComplex operator*(const cuComplex& a) 
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	__device__ cuComplex operator+(const cuComplex& a) 
	{
		return cuComplex(r+a.r, i+a.i);
	}
};

__device__ int juliaGPU(int x, int y)
{
	const float scale = 1.5;
	float jx = scale * (float)(DIM/2 - x)/(DIM/2);
	float jy = scale * (float)(DIM/2 - y)/(DIM/2);
	
	cuComplex c(-0.8, 0.156);
	cuComplex a(jx, jy);
	
	for (int i = 0; i < 200; i++) 
	{
		a = a * a + c;
		if (a.magnitude2() > 1000)
		return 0;
	}
	return 1;
}


/***************************************************/
/************** SECCION PARALELA *******************/
/***************************************************/
__global__ void KernelGPUJulia(unsigned char *imgIn, unsigned char *imgOut, int width, int height)
{
	unsigned int row = blockIdx.y;//*blockDim.y+threadIdx.y;
    unsigned int col = blockIdx.x;//*blockDim.x+threadIdx.x;
    int offset = col + row * DIM;

    // Calculamos el valor de la posicion
    int juliaValue = juliaGPU(col, row);
    //imgIn[offset] = 255 * juliaValue;
    imgOut[offset] = 255 * juliaValue;
}


// Funcion que llama Multiplicacion Kernel sin Tiles
void JuliaKernel(Mat imagen, unsigned char *imgInput, unsigned char *imgOutput, int ancho, int alto)
{
	//variables para la GPU
	int tam_bytes =  sizeof(unsigned char)*ancho*alto*imagen.channels();
	unsigned char *d_Input, *d_Output;		// Para la GPU "device"

	//Reservo Memoria en el dispositivo
	cudaMalloc((void**)&d_Input,tam_bytes);
	cudaMalloc((void**)&d_Output,tam_bytes);

	//Copio los datos al dispositivo
	cudaMemcpy(d_Input,imgInput,tam_bytes,cudaMemcpyHostToDevice);

	// Ejecuto el Kernel (del dispositivo)
	float Blocksize = BLOCK_SIZE;
	dim3 dimBlock(Blocksize, Blocksize,1);
	dim3 dimGrid(ceil((float)ancho/dimBlock.x), ceil((float)alto/dimBlock.y),1);
	
	KernelGPUJulia<<< dimGrid, dimBlock >>>(d_Input, d_Output, ancho, alto);	

	cudaDeviceSynchronize();

	cudaMemcpy (imgOutput,d_Output,tam_bytes,cudaMemcpyDeviceToHost);

	cudaFree(d_Input);
	cudaFree(d_Output);
}

/*************************************************/
/*************FUNCION MAIN ***********************/
/*************************************************/

int main()
{
	unsigned char *imageInput, *imageOutput;

	Mat image (DIM, DIM, CV_8UC1, Scalar(255));

	if(!image.data)
	{
		printf("!!No se pudo cargar la Imagen!! \n");
		return -1;
	}

	Size s = image.size();
	int width = s.width;
	int height = s.height;
	int size = sizeof(unsigned char) * width * height * image.channels();

	imageInput = (unsigned char*)malloc(size);
	imageOutput = (unsigned char*)malloc(size);

	imageInput = image.data;

	cout << "El alto y ancho de la imagen tiene respectivamente " << width << " pixels por " << height << " pixels\n";
	
	JuliaKernel(image, imageInput, imageOutput, width, height);
	//juliaCPU(imageInput);
	
	Mat imageFractal;
	imageFractal.create(DIM,DIM,CV_8UC1);
  	imageFractal.data = imageOutput;
  	imwrite("./outputs/1088273734.png", imageFractal);
}