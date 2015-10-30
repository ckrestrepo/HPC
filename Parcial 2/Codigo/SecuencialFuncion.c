#include<iostream>
#include<cstdlib>
#include<cstdlib>
#include<cuda.h>
#include<highgui.h>
#include<cv.h>

using namespace std;
using namespace cv;


int main()
{
  // variables de tiempo
  clock_t start, finish; 
  double tiempoSecuencial;

  Mat image;
  image = imread("inputs/img1.jpg", 0);   // El cero significa que carga la imagen en escala de grises
  Size s = image.size();
  int Row = s.width;
  int Col = s.height;

  unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*Row*Col*image.channels());

  if( !image.data )
  {
    cout<<"Problemas cargando la Imagen"<<endl;
    return -1;
  }

  img = image.data;

  cout<<"... Secuencial ...\n"<<endl;
  double promedio = 0, temp = 0;
  for (int i = 1; i <= 20; ++i)
  {
    cout <<"Iteracion numero: " << i <<endl;
    Mat imageOutput;
    
    start = clock();
    
    Sobel(image, imageOutput, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    
    finish = clock();

    imwrite("./outputs/1088273734.png", imageOutput);

    tiempoSecuencial = (((double) (finish - start)) / CLOCKS_PER_SEC );
    cout<< "El proceso secuencial tomo: " << tiempoSecuencial << " en ejecutar\n "<< endl;
    temp = temp + tiempoSecuencial;
  }
  promedio = temp / 20;
  cout <<"El promedio de tiempo es de: " <<promedio << endl;
  return 0;
}