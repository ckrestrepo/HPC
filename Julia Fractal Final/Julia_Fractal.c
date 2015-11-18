#include <iostream>
#include <cstdlib>
#include <cstdlib>
#include <cuda.h>
#include <highgui.h>
#include <cv.h>

using namespace std;
using namespace cv;

struct cuComplex 
{
	float r;
	float i;
	cuComplex( float a, float b ) : r(a), i(b) 
	{

	}

	float magnitude2( void ) 
	{ 
		return r * r + i * i; 
	}

	cuComplex operator*(const cuComplex& a) 
	{
		return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
	}
	cuComplex operator+(const cuComplex& a) 
	{
		return cuComplex(r+a.r, i+a.i);
	}
};





int julia(int x, int y)
{
	// El 580 es la dimension en X y en Y (alto y ancho)
	const float scale = 1.5;
	float jx = scale * (float)(580/2 - x)/(580/2);
	float jy = scale * (float)(580/2 - y)/(580/2);
	
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

int main()
{
	unsigned char *imageInput;

	Mat image;
	
	image = imread("./inputs/img1.jpg", 0);

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

	imageInput = image.data;

	cout << "El alto y ancho de la imagen tiene respectivamente " << width << " pixels por " << height << " pixels\n";

	for (int i = 0; i < width; ++i)
	{
		for (int j = 0; j < height; ++j)
		{
			int offset = j + i*width;
			int juliaValue = julia (j, i);
			imageInput[offset*4 + 0] = 255 * juliaValue;
			imageInput[offset*4 + 1] = 0;
			imageInput[offset*4 + 2] = 0;
			imageInput[offset*4 + 3] = 255;
		}
	}

	Mat imageFractal;
  	imageFractal.data = imageInput;

  	imwrite("./outputs/1088273734.png", imageFractal);
}
