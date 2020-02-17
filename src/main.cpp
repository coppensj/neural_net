// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"
#include "mnist_loader.h"

int main(){

    /* float **test_images, **training_images; */
    /* int *test_labels, *training_labels; */
    int num_test_images, num_training_images;
    int test_image_size, training_image_size;
    
    load_data("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte", num_test_images, test_image_size);
    load_data("../data/train-images-idx3-ubyte", "../data/train-labels-idx1-ubyte", num_training_images, training_image_size);

    int N = 3;
    int layer_sizes[N] = {2, 3, 1};
    Network net(layer_sizes, N);

    return 0;
}
