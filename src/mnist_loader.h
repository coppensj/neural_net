// 'mnist_loader.h' SRC: modified from https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

#ifndef MNISTLOADER 
#define MNISTLOADER

#include <string>
#include <fstream>
#include <stdexcept> // std::runtime_error

struct image {
    int 

float** read_mnist_images(std::string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        float** _dataset = new float*[number_of_images];
        uchar *_temp = new uchar[image_size];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new float[image_size];
            file.read((char *)_temp, image_size);
            for(int j=0; j<image_size; j++)
                _dataset[i][j] = float(_temp[j]);
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

int* read_mnist_labels(std::string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);
        
        int* _dataset = new int[number_of_labels];
        uchar _temp;
        for(int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_temp, 1);
            _dataset[i] = _temp;
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

void load_data(std::string image_path, std::string label_path, int& number_of_images, int& image_size) {
    float **images = read_mnist_images(image_path, number_of_images, image_size);
    int *labels = read_mnist_labels(label_path, number_of_images);
    
    std::cout << number_of_images << std::endl;
    std::cout << image_size << std::endl;

    std::cout << "==================" << std::endl;
    std::cout << labels[2] << std::endl;
    std::cout << "==================" << std::endl;

    for(int row=0; row<28; row++){
        for(int col=0; col<28; col++){
            std::cout << images[2][col + row * 28] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "==================" << std::endl;

}

#endif
