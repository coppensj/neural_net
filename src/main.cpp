// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"
#include "mnist_loader.h"

int main() {

    int num_test_images, num_training_images, num_validation_images;
    int image_size;
    
    auto [test_data, training_data, validation_data] = load_data_wrapper(num_test_images, num_training_images, num_validation_images, image_size);
    
    int N = 3;
    int layer_sizes[N] = {2, 3, 1};
    NeuralNetwork net(layer_sizes, N);

    net.SGD(training_data, 30, 10, 3.0);
    net.SGD(training_data, 30, 10, 3.0, test_data);
    
    /* int idx = 0; */
    /* std::cout << "\n===============\n"; */
    /* std::cout << test_data[idx].value << std::endl; */
    /* std::cout << num_test_images << std::endl; */
    /* std::cout << "===============\n"; */
    /* std::cout << training_data[idx].value.transpose() << std::endl; */
    /* std::cout << num_training_images << std::endl; */
    /* std::cout << "===============\n"; */
    /* std::cout << validation_data[idx].value << std::endl; */
    /* std::cout << num_validation_images << std::endl; */

    return 0;
}
