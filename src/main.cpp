// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"
#include "mnist_loader.h"

int main() {

    int num_test_images, num_training_images, num_validation_images;
    int image_size;
    
    auto [test_data, training_data, validation_data] = load_data_wrapper(num_test_images, num_training_images, num_validation_images, image_size);
    
    int N = 3;
    int layer_sizes[N] = {784, 30, 10};
    NeuralNetwork net(layer_sizes, N);

    // data, epochs, batch size, eta
    net.SGD(training_data, 30, 10, 3.0, test_data);
    /* net.SGD(training_data, 2, 10, 3.0, test_data); */
    
    /////////// THIS IS FOR TESTING /////////////
    int idx = 0;
    std::cout << "\n===============\n";
    std::cout << test_data[idx].value << std::endl;
    for (int i=0; i<28; i++) {
        for (int j=0; j<28; j++) 
            printf("%3.0f ", test_data[idx].pixels(j+i*28,0));
        std::cout << std::endl;
    }
    /* std::cout << "===============\n"; */
    /* std::cout << training_data[idx].value.transpose() << std::endl; */
    /* for (int i=0; i<28; i++) { */
    /*     for (int j=0; j<28; j++) */ 
    /*         printf("%3.0f ", training_data[idx].pixels(j+i*28,0)); */
    /*     std::cout << std::endl; */
    /* } */
    /* std::cout << "===============\n"; */
    /* std::cout << validation_data[idx].value << std::endl; */
    /* std::cout << num_validation_images << std::endl; */
    /////////// THIS IS FOR TESTING /////////////

    return 0;
}
