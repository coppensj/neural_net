/*
    Testing code for different neural network configurations.
    Adapted for C++17

    Usage in shell:
        make & ./test_network

    Network (network.cpp) parameters:
        2nd param is epochs count
        3rd param is batch size
        4th param is learning rate (eta)

    Author:
        Jarod Coppens, 2019
*/

#include "network.h"
#include "mnist_loader.h"

int main() {

    int num_test_images, num_training_images, num_validation_images;
    int image_size;
    
    auto [test_data, training_data, validation_data] = LoadDataWrapper(num_test_images, num_training_images, num_validation_images, image_size);
    
    int N = 3;
    int layer_sizes[N] = {784, 30, 10};
    NeuralNetwork net(layer_sizes, N);

    net.SGD(training_data, 30, 10, 3.0, test_data);

    return 0;
}
