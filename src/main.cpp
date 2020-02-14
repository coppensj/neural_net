// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"
#include "mnist_loader.h"

int main(){

    unsigned char **test_images, **training_images;
    unsigned char *test_labels, *training_labels;
    int num_test_images, num_test_labels, num_training_images, num_training_labels;
    int test_image_size, training_image_size;
    
    test_images = read_mnist_images("../data/t10k-images-idx3-ubyte", num_test_images, test_image_size);
    test_labels = read_mnist_labels("../data/t10k-labels-idx1-ubyte", num_test_labels);
    training_images = read_mnist_images("../data/train-images-idx3-ubyte", num_training_images, training_image_size);
    training_labels = read_mnist_labels("../data/train-labels-idx1-ubyte", num_training_labels);
   
    std::cout << num_test_images << " " << num_test_labels << std::endl;
    std::cout << test_image_size << std::endl;
    std::cout << num_training_images << " " << num_training_labels << std::endl;
    std::cout << training_image_size << std::endl;
    
    std::cout << "==================" << std::endl;
    std::cout << int(test_labels[0]) << std::endl;
    std::cout << "==================" << std::endl;

    for(int row=0; row<28; row++){
        for(int col=0; col<28; col++){
            std::cout << test_images[0][col + row * 28] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "==================" << std::endl;

    int N = 3;
    int layer_sizes[N] = {2, 3, 1};
    Network net(layer_sizes, N);

    return 0;
}
