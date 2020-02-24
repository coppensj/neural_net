// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"
#include "mnist_loader.h"

int main(){

    int num_test_images, num_training_images, num_validation_images;
    int image_size;
    
    auto [test_data, training_data, validation_data] = load_data_wrapper(num_test_images, num_training_images, num_validation_images, image_size);
    
    int N = 3;
    int layer_sizes[N] = {2, 3, 1};
    Network net(layer_sizes, N);

    return 0;
}
