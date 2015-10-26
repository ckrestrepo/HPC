#include <cmath>
#include <cv.h>
#include <highgui.h>
#include <time.h>
#include <cuda.h>

#define RED 2
#define GREEN 1
#define BLUE 0

using namespace cv;

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

int main(int argc, char ** argv )
{
  // Definicion de variables
  unsigned char *dataRawImage, *d_dataRawImage, *d_imageOutput, *h_imageOutput, *d_sobelOutput;

  // Definicion de Matrices para el eje X y eje Y
  char GX[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};   // Gx
  char GY[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};   // Gy



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
  cudaMalloc((void**)&d_dataRawImage, size);

  h_imageOutput = (unsigned char *)malloc(sizeGray);
  cudaMalloc((void**)&d_imageOutput, sizeGray);

  cudaMalloc((void**)&d_M,sizeof(char)*9);
  cudaMalloc((void**)&d_sobelOutput, sizeGray);
 
  dataRawImage = image.data;
  cudaMemcpy(d_dataRawImage, dataRawImage, size, cudaMemcpyHostToDevice);

  cudaMemcpy(d_M, h_M, sizeof(char)*9, cudaMemcpyHostToDevice);

  /*************************************************************************************/
  /************ Definicion para convertir la imagen a escala de grises *****************/
  /*************************************************************************************/

  int blockSize = 32;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid(ceil(width/float(blockSize)), ceil(height/float(blockSize)), 1);

  img2gray<<<dimGrid, dimBlock>>>(d_dataRawImage, width, height, d_imageOutput);

  cudaDeviceSynchronize();
  
  printf("Width is: %d and height is: %d\n", width, height );

  /************************************/
  /** Llamada al algoritmo secuencial**/
  /************************************/

  void sobelFilterSequential (unsigned char *imageInput, int width, int height, unsigned int maskWidth,\
        char *M,unsigned char *imageOutput)
  {
    myfile.open("conv.txt", ios::out);

    int SUM, sumX, sumY;
    for(int y = 0; y < height ; y++)
    {
      for(int x = 0; x < width ; x++)
      {
        sumX  = 0;
        sumY  = 0;
        //Image Boundaries
        if(y == 0 || y == height -1)
          SUM = 0;
        else if(x == 0 || x == width - 1)
          SUM = 0;
        else
        {
        //Convolution for X
          for(int i = -1; i < maskWidth; i++)
          {
            for(int j = -1; j < maskWidth; j++)
            {
              sumX = sumX + GX[j+1][i+1] * (int)image(x+j,y+i);
            }
          }
        //Convolution for Y
          for(int i = -1; i < maskWidth; i++)
          {
            for(int j = -1; j < maskWidth; j++)
            {
              sumY = sumY + GY[j+1][i+1] * (int)image(x + j, y + i);
            }
          }
          //Edge strength
          SUM = sqrt(pow((double)sumX, 2) + pow((double)sumY, 2));
          //SUM = sumX + sumY;
        }   
        
        if(SUM > 255) SUM = 255;
        if(SUM < 0) SUM = 0;
        //unsigned char newPixel = (255 - (unsigned char)(SUM));
        px[y][x] = SUM;
        myfile << px[y][x] << "\t";
      }
      myfile << "\n";
    }

    myfile.close();
  }




  Mat gray_image;
  gray_image.create(height, width, CV_8UC1);
  gray_image.data = h_imageOutput;

  Mat gray_image_opencv, grad_x, abs_grad_x;
  cvtColor(image, gray_image_opencv, CV_BGR2GRAY);

  convertScaleAbs(grad_x, abs_grad_x);



/////////////////////////////////////////////////////////
  //-------------------------------------------------//
/*
  cout << argv[1] <<endl;
  CImg <unsigned char> image(argv[1]); 

  ofstream myfile;
  myfile.open("pxl.txt", ios::out);
  //Getting the raster data
  int **px;
  px = new  int *[width];
  for(y = 0; y< height; y++)
  {
    px[y] = new  int [height];
    for(x = 0; x<width; x++)
    {
      px[y][x] = image(x,y);
      myfile << (unsigned char)image(x,y) << "\t";
    }
    myfile << "\n";
  }
  myfile.close();
*/
  //Deteccion de bordes utilizando el algoritmo

  
  CImg<unsigned char> oimage(width,height);
  for(int y = 0; y < image.height(); y++)
  {
    for(int x = 0; x<image.width(); x++)
    {
      oimage(x,y) = px[y][x];
    }
  }
  oimage.save_bmp("oima2ge.bmp");
  return 0;
}

