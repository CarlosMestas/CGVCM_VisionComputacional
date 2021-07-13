#include <iostream>
#include "Image.h"
#include "RGB.h"

using namespace std;


// Implementar una clase pixel

int main(){
//    Image<uchar> image;
//    Image<std::vector<uchar>> image;
    Image<RGB<uchar>> image;
    image.read("/home/cmestas/BSDS300-images/BSDS300/images/test/86016.jpg");
    std::cout << "1. Mostrar imagen normal " << std::endl;
    std::cout << "2. Mostrar imagen en escala de grises" << std::endl;
    std::cout << "3. Mostrar histograma " << std::endl;
    std::cout << "4. Mostrar histograma acumulativo " << std::endl;
    std::cout << "5. Mostrar imagen ecualizada con histograma " << std::endl;
    std::cout << "6. Mostrar imagen con paddings " << std::endl;
    std::cout << "7. Mostrar imagen operación de convolución " << std::endl;
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
        std::cout << "\t" << "1. " << std::endl;
        std::cout << "\t" << "|0 0 0|" << std::endl;
        std::cout << "\t" << "|0 0 1|" << std::endl;
        std::cout << "\t" << "|0 0 0|" << std::endl;

    }







    return 0;
}
