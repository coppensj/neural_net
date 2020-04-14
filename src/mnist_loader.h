// 'mnist_loader.h' SRC: modified from https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

#ifndef NEURAL_NET_MNIST_H_ 
#define NEURAL_NET_MNIST_H_

#include <string>
#include <fstream>
#include <stdexcept>    // std::runtime_error
#include <vector>       // std::vector
#include <Eigen/Core>

#include "data_types.h" // 'training_image' and 'image' types

float** 
ReadMnistImages(std::string full_path, int& number_of_images, int& image_size) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if(magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");

        file.read((char *)&number_of_images, sizeof(number_of_images)), 
            number_of_images = reverseInt(number_of_images);
        file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
        file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

        image_size = n_rows * n_cols;

        float** _dataset = new float*[number_of_images];
        uchar *_temp = new uchar[image_size];
        for(int i = 0; i < number_of_images; i++) {
            _dataset[i] = new float[image_size];
            file.read((char *)_temp, image_size);
            for(int j=0; j<image_size; j++)
                _dataset[i][j] = float(_temp[j]) / 255.0;
        }
        return _dataset;
    } else {
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    }
}

int* 
ReadMnistLabels(std::string full_path, int& number_of_labels) {
    auto reverseInt = [](int i) {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    typedef unsigned char uchar;

    std::ifstream file(full_path, std::ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), 
            number_of_labels = reverseInt(number_of_labels);
        
        int* _dataset = new int[number_of_labels];
        uchar _temp;
        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_temp, 1);
            _dataset[i] = _temp;
        }
        return _dataset;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

// return data types defined in 'data_types.h'
std::tuple<std::vector<image>, std::vector<training_image>, std::vector<image>>
LoadDataWrapper(int& n_test_images, int& n_training_images, int& n_validation_images, int& image_size) {

    float **test_images = ReadMnistImages("../data/t10k-images-idx3-ubyte", n_test_images, image_size);
    int    *test_labels = ReadMnistLabels("../data/t10k-labels-idx1-ubyte", n_test_images);
    float **training_images = ReadMnistImages("../data/train-images-idx3-ubyte", n_training_images, image_size);
    int    *training_labels = ReadMnistLabels("../data/train-labels-idx1-ubyte", n_training_images);
    
    n_validation_images = 10000;
    n_training_images -= n_validation_images;

    std::vector<image> test_data;
    for (int i=0; i<n_test_images; i++) {
        test_data.push_back(image());
        test_data[i].value = test_labels[i];
        test_data[i].pixels = Eigen::Map<Eigen::MatrixXf>(test_images[i], image_size, 1);
    }
   
    std::vector<training_image> training_data;
    for (int i=0; i<n_training_images; i++) {
        training_data.push_back(training_image());
        training_data[i].value(training_labels[i],0) = 1;
        training_data[i].pixels = Eigen::Map<Eigen::MatrixXf>(training_images[i], image_size, 1);
    }
   
    // separate last 10000 images from training data to form a validation set
    std::vector<image> validation_data;
    for (int i=0; i<n_validation_images; i++) {
        int j = i + n_training_images;
        validation_data.push_back(image());
        validation_data[i].value = training_labels[j];
        validation_data[i].pixels = Eigen::Map<Eigen::MatrixXf>(training_images[j], image_size, 1);
    }

    return {test_data, training_data, validation_data};
}

#endif  // NEURAL_NET_MNIST_H_ 
