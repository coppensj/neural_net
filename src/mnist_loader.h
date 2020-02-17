// 'mnist_loader.h' SRC: modified from https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c

#ifndef MNISTLOADER 
#define MNISTLOADER

#include <string>
#include <fstream>
#include <stdexcept> // std::runtime_error

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

struct training_image {
    Eigen::MatrixXf pixels;
    Eigen::Matrix<float, 10, 1> value;
};

struct test_image {
    Eigen::MatrixXf pixels;
    int value;
};

std::tuple<test_image*, training_image*, test_image*>
load_data_wrapper(int& n_test_images, int& n_training_images, int& n_validation_images, int& image_size) {

    float **test_images = read_mnist_images("../data/t10k-images-idx3-ubyte", n_test_images, image_size);
    int    *test_labels = read_mnist_labels("../data/t10k-labels-idx1-ubyte", n_test_images);
    float **training_images = read_mnist_images("../data/train-images-idx3-ubyte", n_training_images, image_size);
    int    *training_labels = read_mnist_labels("../data/train-labels-idx1-ubyte", n_training_images);
    n_validation_images = 10000;

    test_image *test_data = new test_image[n_test_images];
    for(int i=0; i<n_test_images; i++){
        test_data[i].value = test_labels[i];
        test_data[i].pixels = Eigen::Map<Eigen::MatrixXf>(test_images[i], image_size, 1);
    }
    
    training_image *training_data = new training_image[n_training_images - 10000];
    for(int i=0; i<n_training_images - 10000; i++){
        training_data[i].value(training_labels[i],0) = 1;
        training_data[i].pixels = Eigen::Map<Eigen::MatrixXf>(training_images[i], image_size, 1);
    }
    
    test_image *validation_data = new test_image[n_validation_images];
    for(int i=0; i<n_validation_images; i++){
        int j = i + n_training_images - 10000;
        validation_data[i].value = training_labels[j];
        validation_data[i].pixels = Eigen::Map<Eigen::MatrixXf>(training_images[j], image_size, 1);
    }
    
    return {test_data, training_data, validation_data};

    /* int idx = 423; */
    /* /1* std::cout << "==================" << std::endl; *1/ */
    /* /1* std::cout << test_labels[idx] << std::endl; *1/ */
    /* /1* std::cout << test_data[idx].value << std::endl; *1/ */
    /* /1* std::cout << "==================" << std::endl; *1/ */
    /* /1* for(int row=0; row<image_size; row++){ *1/ */
    /* /1*     std::cout << "(" << test_images[idx][row] << ","; *1/ */
    /* /1*     std::cout << test_data[idx].pixels(row,0) << ") \n"; *1/ */
    /* /1* } *1/ */
    /* /1* std::cout << "==================" << std::endl; *1/ */
    
    /* /1* std::cout << training_labels[idx] << std::endl; *1/ */
    /* /1* std::cout << training_data[idx].value << std::endl; *1/ */
    /* /1* std::cout << "==================" << std::endl; *1/ */
    /* /1* for(int row=0; row<image_size; row++){ *1/ */
    /* /1*     std::cout << "(" << training_images[idx][row] << ","; *1/ */
    /* /1*     std::cout << training_data[idx].pixels(row,0) << ") \n"; *1/ */
    /* /1* } *1/ */
    /* /1* std::cout << "==================" << std::endl; *1/ */
    
    /* std::cout << "==================" << std::endl; */
    /* std::cout << training_labels[idx + num_training_images - 10000] << std::endl; */
    /* std::cout << validation_data[idx].value << std::endl; */
    /* std::cout << "==================" << std::endl; */
    /* for(int row=0; row<image_size; row++){ */
    /*     std::cout << "(" << training_images[idx + num_training_images - 10000][row] << ","; */
    /*     std::cout << validation_data[idx].pixels(row,0) << ") \n"; */
    /* } */
    /* std::cout << "==================" << std::endl; */

}

#endif
