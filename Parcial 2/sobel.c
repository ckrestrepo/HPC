#include <cmath>
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;

unsigned char clamp(int value)
{
    if(value < 0)
        value = 0;
    else
        if(value > 255)
            value = 255;
    return (unsigned char)value;
}

__global__ void img2gray(unsigned char *imageInput, int width, int height, unsigned char *imageOutput)
{
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;

    if((row < height) && (col < width))
    {
        imageOutput[row*width+col] = imageInput[(row*width+col)*3+RED]*0.299 + imageInput[(row*width+col)*3+GREEN]*0.587 \
                                     + imageInput[(row*width+col)*3+BLUE]*0.114;
    }
}


// Filtro de Sobel
void sobelFilterSequential (unsigned char *imageInput, int width, int height, unsigned int maskWidth, char *Gx, char *Gy, unsigned char *imageOutput)
{
  int SUM, sumX, sumY;
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      sumX = 0;
      sumY = 0;
      // Image Boundaries
      if (y == 0 || y == height - 1)
        SUM = 0;
      else if (x == 0 || x == width - 1)  
        SUM = 0;
      //:::::::::::::::::::::::::::::::://
      else
      {
        // Convolution for X
        for (int i = 0; i < maskWidth; ++i)
        {
          for (int j = 0; j < maskWidth; ++j)
          {                                                                   // i o j
            sumX = sumX + Gx[i * maskWidth + j] * imageInput[(x * width + i) + (y + j)];
          }
        }
        // Convolution for Y
        for (int i = 0; i < maskWidth; ++i)
        {
          for (int j = 0; j < maskWidth; ++j)
          {
            sumY = sumY + Gy[i * maskWidth + j] * imageInput[(x * width + i) + (y + j)];
          }
        }
        // Bordes
        SUM = sqrt(pow((double) sumX, 2) + pow((double) sumY, 2));
      }
      imageOutput[y * width + x] = clamp(SUM);
    }
  }
}

int main()
{
  // Definicion de variables
  unsigned char *dataRawImage, *h_imageOutput, *d_dataRawImage, *d_imageOutput;

  // Definicion de Matrices Horizontal y Vertical
  char GX[] = {-1, 0, 1, 
               -2, 0, 2, 
               -1, 0, 1};   // Gx

  char GY[] = {-1, -2, -1, 
                0,  0,  0, 
                1,  2,  1};   // Gy

  Mat image;
  image = imread("./inputs/img1.jpg", 1);

  if(!image.data)
  {
    printf("!!No se pudo cargar la Imagen!! \n");
    return -1;
  }

  Size s = image.size();

  int width = s.width;
  int height = s.height;
  int size = sizeof(unsigned char) * width * height * image.channels();
  int sizeGray = sizeof(unsigned char) * width * height;

  dataRawImage = (unsigned char*)malloc(size);
  
  cudaMalloc((void**)&d_dataRawImage,size);

  h_imageOutput = (unsigned char *)malloc(sizeGray);
  
  dataRawImage = image.data;
  
  printf("Width is: %d and height is: %d\n", width, height );
  
  cudaMalloc((void**)&d_imageOutput, sizeGray);
 
  cudaMemcpy(d_dataRawImage, dataRawImage, size, cudaMemcpyHostToDevice);

  /*************************************************************************************/
  /************ Definicion para convertir la imagen a escala de grises *****************/
  /*************************************************************************************/

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

  img2gray<<<dimGrid, dimBlock>>>(d_dataRawImage, width, height, d_imageOutput);

  cudaDeviceSynchronize();

  sobelFilterSequential(d_imageOutput, width, height, 3, GX, GY, h_imageOutput);

  Mat gray_image;
  gray_image.create(height, width, CV_8UC1);
  gray_image.data = h_imageOutput;

  Mat gray_image_opencv, grad_x, abs_grad_x;
  cvtColor(image, gray_image_opencv, CV_BGR2GRAY);
  Sobel(gray_image_opencv, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

  imwrite("./outputs/1088273734.png",gray_image);

  cudaFree(d_dataRawImage);
  cudaFree(d_imageOutput);
  
  return 0;
}

