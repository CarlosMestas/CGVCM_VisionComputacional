/*!
   Computación Gráfica, Visión Computacional y Multimedia
   @author: Carlos Alberto Mestas Escarcena
   @author: David Jose Peña Ugarte
   En esta clase main.cpp se tiene el siguiente procedimiento
   Para la ejecución se tienen que ingresar 1 argumentos
   1. La ruta de la imagen

   EJEMPLO DE EJECUCIÓN

   ./executable /home/cmestas/BSDS300-images/BSDS300/images/test/229036.jpg

    NOTAS:
    El funcionamiento es con un menú, este funciona en un ciclo hasta que se solicite que termine
*/

#include <iostream>
#include "Image.h"
#include "RGB.h"

using namespace std;

int main(int argc, char **argv){
//    Image<uchar> image;
//    Image<std::vector<uchar>> image;
    Image<RGB<uchar>> image;
//    image.read("/home/cmestas/BSDS300-images/BSDS300/images/test/86016.jpg");
//    image.read("/home/cmestas/BSDS300-images/BSDS300/images/test/241048.jpg");
//    image.read("/home/cmestas/BSDS300-images/BSDS300/images/test/229036.jpg");

    image.read(argv[1]);

    while(1){
        std::cout << "1.  Mostrar imagen normal" << std::endl;
        std::cout << "2.  Mostrar imagen en escala de grises" << std::endl;
        std::cout << "3.  Mostrar histograma " << std::endl;
        std::cout << "4.  Mostrar histograma acumulativo " << std::endl;
        std::cout << "5.  Mostrar imagen ecualizada con histograma " << std::endl;
        std::cout << "6.  Mostrar imagen con paddings " << std::endl;
        std::cout << "7.  Mostrar imagen operación de convolución " << std::endl;
        std::cout << "8.  Mediana " << std::endl;
        std::cout << "9.  Sobel " << std::endl;
        std::cout << "10. Prewitt " << std::endl;
        std::cout << "99. SALIR" << std::endl;
        int opt;
        std::cout << "Seleccione opción: ";
        std::cin >> opt;
        if(opt == 1)
            image.print();
        else if(opt == 2)
            image.printGrey();
        else if(opt == 3)
            image.histogram();
        else if(opt == 4)
            image.cumulativeHistogram();
        else if(opt == 5)
            image.equalizeImageHistogram();
        else if(opt == 6){
            std::cout << "\t" << "1. Valor constante" << std::endl;
            std::cout << "\t" << "2. Pixeles extendidos" << std::endl;
            std::cout << "\t" << "3. Espejo" << std::endl;
            std::cout << "\t" << "4. Repetición" << std::endl;
            int opt2;
            std::cout << "\t" << "Seleccione opción: ";
            std::cin >> opt2;
            image.padding(opt2);

        }

        else if(opt == 7){
            std::cout << "\t" << "\t" << "1. " << std::endl;
            std::cout << "\t" << "\t" << "|0 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 1|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "2. " << std::endl;
            std::cout << "\t" << "\t" << "|0 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 2 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "3. " << std::endl;
            std::cout << "\t" << "\t" << "|0 0 1|" << std::endl;
            std::cout << "\t" << "\t" << "|0 1 0|" << std::endl;
            std::cout << "\t" << "\t" << "|1 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "4. " << std::endl;
            std::cout << "\t" << "\t" << "|1 0 1 1 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 1 0 1 1|" << std::endl;
            std::cout << "\t" << "\t" << "|1 0 1 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "|1 0 1 0 1|" << std::endl;
            std::cout << "\t" << "\t" << "|1 0 1 0 1|" << std::endl;
            std::cout << "\t" << "\t" << "5. " << std::endl;
            std::cout << "\t" << "\t" << "|0 0 1 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 0 1 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 1 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 1 0 0|" << std::endl;
            std::cout << "\t" << "\t" << "|0 0 1 0 0|" << std::endl;
            int opt3;
            int scale;
            std::cout << "\t" << "\t" << "Seleccione opción: ";
            std::cin >> opt3;
            std::cout << "\t" << "\t" << "Ingrese valor de la escala: ";
            std::cin >> scale;
            std::cout << "\t" << "\t" << "1. Espejo" << std::endl;
            //std::cout << "\t" << "\t" << "2. Repetición" << std::endl;
            int opt4;
            std::cout << "\t" << "\t" << "Seleccione opción: ";
            std::cin >> opt4;
            int **kernel;
            int size;
            if(opt3 == 1 || opt3 == 2 || opt3 == 3){
                size = 3;
            }
            else{
                size = 5;
            }
            kernel = new int*[size];
            for(int i = 0; i < size; ++i){
                kernel[i] = new int[size];
            }
            if(opt3 == 1){
                kernel[0][0] = 0;
                kernel[0][1] = 0;
                kernel[0][2] = 0;
                kernel[1][0] = 0;
                kernel[1][1] = 0;
                kernel[1][2] = 1;
                kernel[2][0] = 0;
                kernel[2][1] = 0;
                kernel[2][2] = 0;
            }
            else if(opt3 == 2){
                kernel[0][0] = 0;
                kernel[0][1] = 0;
                kernel[0][2] = 0;
                kernel[1][0] = 0;
                kernel[1][1] = 2;
                kernel[1][2] = 0;
                kernel[2][0] = 0;
                kernel[2][1] = 0;
                kernel[2][2] = 0;
            }

            else if(opt3 == 3){
                kernel[0][0] = 0;
                kernel[0][1] = 0;
                kernel[0][2] = 1;
                kernel[1][0] = 0;
                kernel[1][1] = 1;
                kernel[1][2] = 0;
                kernel[2][0] = 1;
                kernel[2][1] = 0;
                kernel[2][2] = 0;
            }
            else if(opt3 == 4){
                kernel[0][0] = 1;
                kernel[0][1] = 0;
                kernel[0][2] = 1;
                kernel[0][3] = 1;
                kernel[0][4] = 0;

                kernel[1][0] = 0;
                kernel[1][1] = 1;
                kernel[1][2] = 0;
                kernel[1][3] = 1;
                kernel[1][4] = 1;

                kernel[2][0] = 1;
                kernel[2][1] = 0;
                kernel[2][2] = 1;
                kernel[2][3] = 0;
                kernel[2][4] = 0;

                kernel[3][0] = 1;
                kernel[3][1] = 0;
                kernel[3][2] = 1;
                kernel[3][3] = 0;
                kernel[3][4] = 1;

                kernel[4][0] = 1;
                kernel[4][1] = 0;
                kernel[4][2] = 1;
                kernel[4][3] = 0;
                kernel[4][4] = 1;
            }
            image.convolution(kernel, scale, size, opt4);

        }
         else if(opt == 8){
          int tam = 1;
          image.mediana(tam,3);

        }
         //  image.histogram();
        else if(opt == 9)
            image.sobel();
        else if(opt == 10)
            image.prewitt();
        else{
            break;
        }
    }
    return 0;
}
