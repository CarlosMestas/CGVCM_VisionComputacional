#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <string>

// Agregamos todo el motor de opencv
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

template<typename PixelType>


class Image{
    // Vector de vectores (no importa el tipo de pixel rgb, etc ...)
    using DataType = std::vector<std::vector<PixelType>>;

public:
    Image(){};
    void read(const std::string& _fileName);
    void print();
    void printGrey();
    void histogram();
    void cumulativeHistogram();
    void equalizeImageHistogram();
    void padding(int _opt);
    void convolution(int **_kernel, int _scale, int _size, int _opt);


private:
    cv::Mat image;
    // Matriz de pixeles
    DataType data;
    // Filas
    std::size_t rows{};
    // Columnas
    std::size_t columns{};

    DataType dataGray;

    std::string fileName;

    cv::Mat convolutionInternal(cv::Mat _image, int **_kernel, int _scale, int _size, int _opt);
    cv::Mat convolutionData(DataType _data, int ** _kernel_, int _scale, int _size);
    int convolutionOperation(int ** _subMat, int ** _kernel, int _scale, int _size);

};

template<typename PixelType>
void Image<PixelType>::read(const std::string& _fileName){
    fileName = _fileName;
    // Lectura de imagen
    image = cv::imread(_fileName, cv::IMREAD_COLOR);
    /*
    // Lectura de imagen en escala de grises
    cv::Mat image = cv::imread(_fileName, cv::IMREAD_GRAYSCALE);
    */
    // Verificamos si la lectura fue correcta
    if(!image.data){
        std::cerr<< "No image data" << std::endl;
        return;
    }

    rows = image.rows;
    columns = image.cols;

    // Damos memoria a nuestra matriz
    data = DataType(rows, std::vector<PixelType>(columns, PixelType{}));

    uchar red, green, blue;


    for(unsigned r = 0; r < rows; ++r){

        cv::Vec3b * row = image.ptr<cv::Vec3b>(r);
        for(unsigned c = 0; c < columns; ++c){
            red =   row[c][2];
            green = row[c][1];
            blue =  row[c][0];

            /*
            red   = image.ptr<cv::Vec3b>(r)[c][2];
            green = image.ptr<cv::Vec3b>(r)[c][1];
            blue  = image.ptr<cv::Vec3b>(r)[c][0];
            */

            // Ingresar los valores en la matriz data

            // if en tiempo de compilacion
            // Comprobacion si es un tipo de dato basico int, float, etc ...
            if constexpr (std::is_fundamental<PixelType>::value){
                // Conversion de RGB a grises
                data[r][c] = static_cast<PixelType>((red + green + blue) / 3);
            }
            else{
                data[r][c][0] = static_cast<typename PixelType::valueType>(red);
                data[r][c][1] = static_cast<typename PixelType::valueType>(green);
                data[r][c][2] = static_cast<typename PixelType::valueType>(blue);
            }
        }
    }
}

template<typename PixelType>
void Image<PixelType>::print(){

    cv::Mat image = cv::imread(fileName, cv::IMREAD_COLOR);

    if (!image.data)
    {
        std::cerr<<"No image data\n";
        return;
    }

    cv::namedWindow("Imagen a colores", cv::WINDOW_AUTOSIZE);
    cv::imshow("Imagen a colores", image);
    cv::waitKey(0);


}

template<typename PixelType>
void Image<PixelType>::printGrey(){

    uint8_t greyImage[rows][columns];


///    std::cout << "ga" << std::endl;
    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
//           std::cout << +data[r][c][0] << ", " << +data[r][c][1] << ", " << +data[r][c][2] << std::endl;
            greyImage[r][c] = (uint8_t)(((static_cast<int>(data[r][c][0])) + (static_cast<int>(data[r][c][1])) + (static_cast<int>(data[r][c][2])))/3);
//            std::cout << "v " << r << " - " << c << " " << ((static_cast<int>(data[r][c][0])) + (static_cast<int>(data[r][c][1])) + (static_cast<int>(data[r][c][2])))/3 <<std::endl;
//            std::cout << "gi " << static_cast<int>(greyImage[r][c]) << std::endl;
        }
//        std::cout << std::endl;
    }

    cv::Mat greyImg = cv::Mat(rows, columns, CV_8U, &greyImage);
    std::string greyArrWindow = "Imagen en escala de grises";
    cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    cv::imshow(greyArrWindow, greyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

}

template<typename PixelType>
void Image<PixelType>::histogram(){

    uint8_t greyImage[rows][columns];

    int histogramData[255];
    int histogramDataPrint[255];

    for(unsigned i = 0; i < 255; ++i){
        histogramData[i] = 0;
    }

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
            uint8_t value = (uint8_t)(((static_cast<int>(data[r][c][0])) + (static_cast<int>(data[r][c][1])) + (static_cast<int>(data[r][c][2])))/3);
            greyImage[r][c] = value;
            histogramData[(int)greyImage[r][c]] = histogramData[(int)greyImage[r][c]] + 1;
        }
    }

    for(unsigned i = 0; i < 255; ++i){
        std::cout << i << "\t" << histogramData[i] << std::endl;
    }

    int maxValue = -1;


    for(unsigned i = 0; i < 255; ++i){
        histogramDataPrint[i] = histogramData[i] / 6;
    }

    for(unsigned i = 0; i < 255; ++i){
        if(maxValue < histogramDataPrint[i]){
            maxValue = histogramDataPrint[i];
        }
    }

    std::cout << "max value " << maxValue << std::endl;

    int r = 2;
    int columnsSize = 255 * r;
    uint8_t histogramImage[maxValue][columnsSize];

    int q = 0;
    for(unsigned j = 0; j < columnsSize ; j = j + r){

        int k = histogramDataPrint[q];
        for(unsigned i = 0; i < maxValue; i++){

            if(k > 0){
                histogramImage[maxValue - i][(j)] = 0;
                histogramImage[maxValue - i][(j+1)] = 0;
            }
            else{
                histogramImage[maxValue - i][(j)] = 150;
                histogramImage[maxValue - i][(j+1)] = 150;
            }
            k--;
        }
        q++;
    }

    cv::Mat greyImg = cv::Mat(maxValue, columnsSize, CV_8U, &histogramImage);
    std::string greyArrWindow = "Histograma de la imagen (grises)";
    cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    cv::imshow(greyArrWindow, greyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

}

template<typename PixelType>
void Image<PixelType>::cumulativeHistogram(){

    uint8_t greyImage[rows][columns];

    int histogramData[255];
    int histogramDataPrint[255];
    int cumulativeHistogram[255];

    for(unsigned i = 0; i < 255; ++i){
        histogramData[i] = 0;
    }

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
            uint8_t value = (uint8_t)(((static_cast<int>(data[r][c][0])) + (static_cast<int>(data[r][c][1])) + (static_cast<int>(data[r][c][2])))/3);
            greyImage[r][c] = value;
            histogramData[(int)greyImage[r][c]] = histogramData[(int)greyImage[r][c]] + 1;
        }
    }

    for(unsigned i = 0; i < 255; ++i){
        std::cout << i << "\t" << histogramData[i] << std::endl;
    }

    int maxValue = -1;



    cumulativeHistogram[0] = histogramData[0];
    for(unsigned i = 0; i < 254; ++i){
        cumulativeHistogram[i+1] = cumulativeHistogram[i] + histogramData[i+1];
    }

    for(unsigned i = 0; i < 255; ++i){
        cumulativeHistogram[i] = cumulativeHistogram[i] / 500;
    }


    for(unsigned i = 0; i < 255; ++i){
        if(maxValue < cumulativeHistogram[i]){
            maxValue = cumulativeHistogram[i];
        }
    }

    std::cout << "max value " << maxValue << std::endl;

    int r = 2;
    int columnsSize = 255 * r;
    uint8_t histogramImage[maxValue][columnsSize];

    int q = 0;
    for(unsigned j = 0; j < columnsSize ; j = j + r){

        int k = cumulativeHistogram[q];
        for(unsigned i = 0; i < maxValue; i++){

            if(k > 0){
                histogramImage[maxValue- i][(j)] = 0;
                histogramImage[maxValue- i][(j+1)] = 0;
            }
            else{
                histogramImage[maxValue- i][(j)] = 150;
                histogramImage[maxValue- i][(j+1)] = 150;
            }
            k--;
        }
        q++;
    }

    cv::Mat greyImg = cv::Mat(maxValue, columnsSize, CV_8U, &histogramImage);
    std::string greyArrWindow = "Histograma de la imagen ecualizado (grises)";
    cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    cv::imshow(greyArrWindow, greyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

}


template<typename PixelType>
void Image<PixelType>::equalizeImageHistogram(){

    uint8_t greyImage[rows][columns];
    uint8_t greyImageEqualize[rows][columns];

    int histogramData[255];
    int histogramDataPrint[255];
    int cumulativeHistogram[255];

    for(unsigned i = 0; i < 255; ++i){
        histogramData[i] = 0;
    }

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
            uint8_t value = (uint8_t)(((static_cast<int>(data[r][c][0])) + (static_cast<int>(data[r][c][1])) + (static_cast<int>(data[r][c][2])))/3);
            greyImage[r][c] = value;
            histogramData[(int)greyImage[r][c]] = histogramData[(int)greyImage[r][c]] + 1;
        }
    }

    for(unsigned i = 0; i < 255; ++i){
        std::cout << i << "\t" << histogramData[i] << std::endl;
    }

    int maxValue = -1;

    cumulativeHistogram[0] = histogramData[0];
    for(unsigned i = 0; i < 254; ++i){
        cumulativeHistogram[i+1] = cumulativeHistogram[i] + histogramData[i+1];
    }

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
            int a = static_cast<int>(greyImage[r][c]);
            int b = cumulativeHistogram[a] * (256 - 1)/ (rows * columns);
            greyImageEqualize[r][c] = (uint8_t)b;

        }
    }

    cv::Mat greyImg = cv::Mat(rows, columns, CV_8U, &greyImageEqualize);
    std::string greyArrWindow = "Imagen en escala de grises ecualizado";
    cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    cv::imshow(greyArrWindow, greyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

}

template<typename PixelType>
void Image<PixelType>::padding(int _opt){

    uint8_t greyImage[rows][columns];
    int sizeKernel = 101;
    int newPixels = sizeKernel / 2;

    int newRows = rows + newPixels * 2;
    int newColumns = columns + newPixels * 2;

    uint8_t greyImagePadding[newRows][newColumns];

    std::string name = "";

    for(int r = 0; r < rows; r++){
        for(int c = 0; c < columns; c++){
            uint8_t value = (uint8_t)(((static_cast<int>(data[r][c][0])) + (static_cast<int>(data[r][c][1])) + (static_cast<int>(data[r][c][2])))/3);
            greyImage[r][c] = value;
        }
    }

    if(_opt == 1){
        name = "Imagen en escala de grises - padding valor constante";
        for(int r = 0; r < newRows; r++){
            for(int c = 0; c < newColumns; c++){
                uint8_t value = (uint8_t) 127;
                greyImagePadding[r][c] = value;
            }
        }
        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t value = greyImage[r][c];
                greyImagePadding[r + newPixels][c + newPixels] = value;
            }
        }

    }

    else if(_opt == 2){
        name = "Imagen en escala de grises - padding extención de pixeles";
        for(int r = 0; r < newRows; r++){
            for(int c = 0; c < newColumns; c++){
                uint8_t value = (uint8_t) 127;
                greyImagePadding[r][c] = value;
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t value = greyImage[r][c];
                greyImagePadding[r + newPixels][c + newPixels] = value;
            }
        }

        for(int r = 0; r < rows; r++){
            uint8_t value = greyImage[r][0];
            for(int c = c + newPixels; c > -1; c--){
                greyImagePadding[r + newPixels ][c] = value;
            }
        }

        for(int r = 0; r < rows; r++){
            uint8_t value = greyImage[r][columns - 1];
            for(int c = columns + newPixels; c < columns + newPixels *2; c++){
                greyImagePadding[r + newPixels][c] = value;
            }
        }


        /////////////////////////////////
        for(int c = 0; c < columns; c++){
            uint8_t value = greyImage[0][c];
            for(int r = newPixels - 1; r > -1; r--){
                greyImagePadding[r][c + newPixels] = value;
            }
        }

        for(int c = 0; c < columns; c++){
            uint8_t value = greyImage[rows - 1][c];
            for(int r = rows + newPixels; r < rows + newPixels * 2; r++){
                greyImagePadding[r][c + newPixels] = value;
            }
        }
        /*

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t value = greyImage[r][c];
                greyImagePadding[r + newPixels][c + newPixels] = value;
            }
        }


        */

        uint8_t d00 = greyImage[0][0];
        uint8_t d01 = greyImage[0][columns - 1];
        uint8_t d02 = greyImage[rows - 1][0];
        uint8_t d03 = greyImage[rows - 1][columns - 1];


        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                greyImagePadding[r][c] = d00;
                greyImagePadding[r][c + columns + newPixels] = d01;
                greyImagePadding[r + rows + newPixels][c] = d02;
                greyImagePadding[r + rows + newPixels][c + columns + newPixels] = d03;
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t value = greyImage[r][c];
                greyImagePadding[r + newPixels][c + newPixels] = value;
            }
        }
    }

    else if (_opt == 3){
        name = "Imagen en escala de grises - padding espejo";
        for(int r = 0; r < newRows; r++){
            for(int c = 0; c < newColumns; c++){
                uint8_t value = (uint8_t) 127;
                greyImagePadding[r][c] = value;
            }
        }


        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t value = greyImage[r][c];
                greyImagePadding[r + newPixels][c + newPixels] = value;
            }
        }


        uint8_t left[rows][newPixels];

    //    std::cout << rows << "\t" << newPixels << std::endl;

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                left[r][c] = greyImage[r][c];
            }
        }

       // greyImagePadding[0+newPixels/2][0] = 255;
        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels ; c++){
                greyImagePadding[r + newPixels][c] = left[r][newPixels - 1 - c];
//                greyImagePadding[r + newPixels][c] = left[r][c];

            }
        }

        uint8_t right[rows][newPixels];

        for(int r = 0; r < rows; r++){
//            for(int c = columns - newPixels; c < columns; c++){
            for(int c = 0; c < newPixels; c++){
                right[r][c] = greyImage[r][columns - 1 - c];
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                greyImagePadding[r + newPixels][c + columns + newPixels] = right[r][c];
            //for (int c = 0; c < newPixels; c++){
            //    greyImagePadding[r + newPixels][c + columns + newPixels] = right[r][newPixels - c];
//                greyImagePadding[r + newPixels][c + columns + newPixels] = right[r][c];

            }
        }


        /*
        for(int r = 0; r < rows; r++){
            uint8_t value = greyImage[r][0];
            for(int c = c + newPixels; c > -1; c--){
                greyImagePadding[r + newPixels ][c] = value;
            }
        }
        */
        uint8_t up[newPixels][columns];
        uint8_t down[newPixels][columns];

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < columns; c++){
                up[r][c]    = greyImage[r][c];
                down[r][c]  = greyImage[rows - 1 - r][c];
            }
        }

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < columns; c++){
                greyImagePadding[r][c + newPixels] = up[newPixels - 1 - r][c];
                greyImagePadding[r + rows + newPixels][c + newPixels] = down[r][c];
            }
        }


        uint8_t d01[newPixels][newPixels];
        uint8_t d02[newPixels][newPixels];
        uint8_t d03[newPixels][newPixels];
        uint8_t d04[newPixels][newPixels];

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                d01[r][c] = greyImage[r][c];
                d02[r][c] = greyImage[r][columns - 1 - c];
                d03[r][c] = greyImage[rows - 1 - r][c];
                d04[r][c] = greyImage[rows - 1 - r][columns - 1 - c];
            }
        }

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                greyImagePadding[r][c] = d01[newPixels - 1 - r][newPixels - 1 - c];
                greyImagePadding[r][c + columns + newPixels] = d02[newPixels - 1 - r][c];
                greyImagePadding[r + rows + newPixels][c] = d03[r][newPixels - 1 - c];
                greyImagePadding[r + rows + newPixels][c + columns + newPixels] = d04[r][c];

            }
        }

    }

    else if (_opt == 4){
        name = "Imagen en escala de grises - padding repetición";
        for(int r = 0; r < newRows; r++){
            for(int c = 0; c < newColumns; c++){
                uint8_t value = (uint8_t) 127;
                greyImagePadding[r][c] = value;
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t value = greyImage[r][c];
                greyImagePadding[r + newPixels][c + newPixels] = value;
            }
        }

        uint8_t left[rows][newPixels];
        uint8_t right[rows][newPixels];

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                left[r][c] = greyImage[r][c];
                right[r][c] = greyImage[r][columns - 1 - c];
            }
        }

        uint8_t up[newPixels][columns];
        uint8_t down[newPixels][columns];

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < columns; c++){
                up[r][c]    = greyImage[r][c];
                down[r][c]  = greyImage[rows - 1 - r][c];
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels ; c++){
                greyImagePadding[r + newPixels][c] = right[r][newPixels - 1 - c];
                greyImagePadding[r + newPixels][c + columns + newPixels] = left[r][c];
            }
        }

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < columns; c++){
                greyImagePadding[r][c + newPixels] = down[newPixels - 1 - r][c];
                greyImagePadding[r + rows + newPixels][c + newPixels] = up[r][c];
            }
        }

        uint8_t d01[newPixels][newPixels];
        uint8_t d02[newPixels][newPixels];
        uint8_t d03[newPixels][newPixels];
        uint8_t d04[newPixels][newPixels];

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                d01[r][c] = greyImage[r][c];
                d02[r][c] = greyImage[r][columns - 1 - c];
                d03[r][c] = greyImage[rows - 1 - r][c];
                d04[r][c] = greyImage[rows - 1 - r][columns - 1 - c];
            }
        }

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                greyImagePadding[r][c] = d04[newPixels - 1 - r][newPixels - 1 - c];
                greyImagePadding[r][c + columns + newPixels] = d03[newPixels - 1 - r][c];
                greyImagePadding[r + rows + newPixels][c] = d02[r][newPixels - 1 - c];
                greyImagePadding[r + rows + newPixels][c + columns + newPixels] = d01[r][c];

            }
        }

    }

    cv::Mat greyImg = cv::Mat(newRows, newColumns, CV_8U, &greyImagePadding);
    std::string greyArrWindow = name;
    cv::namedWindow(greyArrWindow, cv::WINDOW_AUTOSIZE);
    cv::imshow(greyArrWindow, greyImg);
    cv::waitKey(0);
    cv::destroyAllWindows();

}


template<typename PixelType>
int Image<PixelType>::convolutionOperation(int ** _subMat, int ** _kernel, int _scale, int _size){

    return 0;
}

template<typename PixelType>
cv::Mat Image<PixelType>::convolutionData(DataType _data, int ** _kernel, int _scale, int _size){
    cv::Mat newImage = image.clone();
    DataType dataCon = DataType((rows), std::vector<PixelType>((columns), PixelType{}));

    int valueExtremeRight = _size / 2;
    int valueExtremeLeft = (-1) * valueExtremeRight;

    for(unsigned r = 0; r < rows; ++r){

        cv::Vec3b * row = newImage.ptr<cv::Vec3b>(r);
        for(unsigned c = 0; c < columns; ++c){

            int finalValueR = 0;
            int finalValueG = 0;
            int finalValueB = 0;


            for(unsigned i = 0; i < _size; ++i){
                for(unsigned j = 0; j < _size; ++j){
                    finalValueR += (_kernel[i][j] * (int)_data[r + i][c + j][0]);
                    finalValueG += (_kernel[i][j] * (int)_data[r + i][c + j][1]);
                    finalValueB += (_kernel[i][j] * (int)_data[r + i][c + j][2]);
                }
            }
            finalValueR = finalValueR / _scale;
            finalValueG = finalValueG / _scale;
            finalValueB = finalValueB / _scale;

            if(finalValueR > 255)
                finalValueR = 255;
            if(finalValueR < 0)
                finalValueR = 0;
            if(finalValueG > 255)
                finalValueG = 255;
            if(finalValueG < 0)
                finalValueG = 0;
            if(finalValueB > 255)
                finalValueB = 255;
            if(finalValueB < 0)
                finalValueB = 0;

            row[c][2] = finalValueR;
            row[c][1] = finalValueG;
            row[c][0] = finalValueB;
        }
    }



    return newImage;
}


template<typename PixelType>
cv::Mat Image<PixelType>::convolutionInternal(cv::Mat _image, int **_kernel, int _scale, int _size, int _opt){
    cv::Mat imageClone = _image.clone();
    cv::Mat imageConvu;
    int newPixels = _size / 2;
    int newRows = newPixels * 2;
    int newColumns = newPixels * 2;
    DataType datacopy = DataType((rows + newRows), std::vector<PixelType>((columns + newColumns), PixelType{}));

    // Padding espejo
    if(_opt == 1){
        for(int r = 0; r < rows + newRows; r++){
            for(int c = 0; c < columns + newColumns; c++){
                uint8_t value = (uint8_t) 127;
                datacopy[r][c][0] = (u_int8_t)value; // r
                datacopy[r][c][1] = value; // g
                datacopy[r][c][2] = value; // b
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < columns; c++){
                uint8_t valueR = data[r][c][0];
                uint8_t valueG = data[r][c][1];
                uint8_t valueB = data[r][c][2];
                datacopy[r + newPixels][c + newPixels][0] = valueR;
                datacopy[r + newPixels][c + newPixels][1] = valueG;
                datacopy[r + newPixels][c + newPixels][2] = valueB;

            }
        }

        DataType dataLeft = DataType((rows), std::vector<PixelType>((newPixels), PixelType{}));

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                dataLeft[r][c][0] = data[r][c][0];
                dataLeft[r][c][1] = data[r][c][1];
                dataLeft[r][c][2] = data[r][c][2];
            }
        }


        DataType dataRight = DataType((rows), std::vector<PixelType>((newPixels), PixelType{}));

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                dataRight[r][c][0] = data[r][columns - 1 - c][0];
                dataRight[r][c][1] = data[r][columns - 1 - c][1];
                dataRight[r][c][2] = data[r][columns - 1 - c][2];
            }
        }

        DataType dataUp = DataType((newPixels), std::vector<PixelType>((columns), PixelType{}));
        DataType dataDown = DataType((newPixels), std::vector<PixelType>((columns), PixelType{}));

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < columns; c++){
                dataUp[r][c][0]    = data[r][c][0];
                dataUp[r][c][1]    = data[r][c][1];
                dataUp[r][c][2]    = data[r][c][2];

                dataDown[r][c][0]  = data[rows - 1 - r][c][0];
                dataDown[r][c][1]  = data[rows - 1 - r][c][1];
                dataDown[r][c][2]  = data[rows - 1 - r][c][2];
            }
        }

        DataType dataD01 = DataType((newPixels), std::vector<PixelType>((newPixels), PixelType{}));
        DataType dataD02 = DataType((newPixels), std::vector<PixelType>((newPixels), PixelType{}));
        DataType dataD03 = DataType((newPixels), std::vector<PixelType>((newPixels), PixelType{}));
        DataType dataD04 = DataType((newPixels), std::vector<PixelType>((newPixels), PixelType{}));


        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                dataD01[r][c][0] = data[r][c][0];
                dataD01[r][c][1] = data[r][c][1];
                dataD01[r][c][2] = data[r][c][2];

                dataD02[r][c][0] = data[r][columns - 1 - c][0];
                dataD02[r][c][1] = data[r][columns - 1 - c][1];
                dataD02[r][c][2] = data[r][columns - 1 - c][2];

                dataD03[r][c][0] = data[rows - 1 - r][c][0];
                dataD03[r][c][1] = data[rows - 1 - r][c][1];
                dataD03[r][c][2] = data[rows - 1 - r][c][2];

                dataD04[r][c][0] = data[rows - 1 - r][columns - 1 - c][0];
                dataD04[r][c][1] = data[rows - 1 - r][columns - 1 - c][1];
                dataD04[r][c][2] = data[rows - 1 - r][columns - 1 - c][2];
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                datacopy[r + newPixels][c][0] = dataLeft[r][newPixels - 1 - c][0];
                datacopy[r + newPixels][c][1] = dataLeft[r][newPixels - 1 - c][1];
                datacopy[r + newPixels][c][2] = dataLeft[r][newPixels - 1 - c][2];
            }
        }

        for(int r = 0; r < rows; r++){
            for(int c = 0; c < newPixels; c++){
                datacopy[r + newPixels][c + columns + newPixels][0] = dataRight[r][c][0];
                datacopy[r + newPixels][c + columns + newPixels][1] = dataRight[r][c][1];
                datacopy[r + newPixels][c + columns + newPixels][2] = dataRight[r][c][2];
            }
        }

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < columns; c++){
                datacopy[r][c + newPixels][0] = dataUp[newPixels - 1 - r][c][0];
                datacopy[r][c + newPixels][1] = dataUp[newPixels - 1 - r][c][1];
                datacopy[r][c + newPixels][2] = dataUp[newPixels - 1 - r][c][2];

                datacopy[r + rows + newPixels][c + newPixels][0] = dataDown[r][c][0];
                datacopy[r + rows + newPixels][c + newPixels][1] = dataDown[r][c][1];
                datacopy[r + rows + newPixels][c + newPixels][2] = dataDown[r][c][2];
            }
        }

        for(int r = 0; r < newPixels; r++){
            for(int c = 0; c < newPixels; c++){
                datacopy[r][c][0] = dataD01[newPixels - 1 - r][newPixels - 1 - c][0];
                datacopy[r][c][1] = dataD01[newPixels - 1 - r][newPixels - 1 - c][1];
                datacopy[r][c][2] = dataD01[newPixels - 1 - r][newPixels - 1 - c][2];

                datacopy[r][c + columns + newPixels][0] = dataD02[newPixels - 1 - r][c][0];
                datacopy[r][c + columns + newPixels][1] = dataD02[newPixels - 1 - r][c][1];
                datacopy[r][c + columns + newPixels][2] = dataD02[newPixels - 1 - r][c][2];

                datacopy[r + rows + newPixels][c][0] = dataD03[r][newPixels - 1 - c][0];
                datacopy[r + rows + newPixels][c][1] = dataD03[r][newPixels - 1 - c][1];
                datacopy[r + rows + newPixels][c][2] = dataD03[r][newPixels - 1 - c][2];

                datacopy[r + rows + newPixels][c + columns + newPixels][0] = dataD04[r][c][0];
                datacopy[r + rows + newPixels][c + columns + newPixels][1] = dataD04[r][c][1];
                datacopy[r + rows + newPixels][c + columns + newPixels][2] = dataD04[r][c][2];
            }
        }

        imageConvu = convolutionData(datacopy, _kernel, _scale, _size);

    }


    /*
    std::cout<<+data[0][0][0]<<","<<+data[0][0][1]<<","<<+data[0][0][2]<<"\n";
    std::cout<<+datacopy[0][0][0]<<","<<+datacopy[0][0][1]<<","<<+datacopy[0][0][2]<<"\n";
    */


    /*
    for(unsigned r = 0; r < rows; ++r){

        cv::Vec3b * row = imageClone.ptr<cv::Vec3b>(r);
        for(unsigned c = 0; c < columns; ++c){
            row[c][2] = +datacopy[r + newPixels][c + newPixels][0];
            row[c][1] = +datacopy[r + newPixels][c + newPixels][1];
            row[c][0] = +datacopy[r + newPixels][c + newPixels][2];
        }
    }
    */
    return imageConvu;



}


template<typename PixelType>
void Image<PixelType>::convolution(int **_kernel, int _scale, int _size, int _opt){
    for(int i = 0; i < _size; ++i){
        for(int j = 0; j < _size; ++j){
            std::cout << _kernel[i][j] << " ";
        }
        std::cout << std::endl;
    }

    cv::Mat finalImage = convolutionInternal(image, _kernel, _scale, _size, _opt);

    cv::namedWindow("Imagen a colores", cv::WINDOW_AUTOSIZE);
    cv::imshow("Imagen a colores", finalImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}



#endif // IMAGE_H
