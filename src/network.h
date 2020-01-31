// 'network.h' SRC: J.Coppens 2020

#ifndef NETWORK
#define NETWORK

#include <random>
#include <iostream>

class Network {
    // Data
    public:
        int num_layers;
        int *sizes;
        std::vector<std::vector<float>> biases;
        std::vector<std::vector<std::vector<float>>> weights;

    // Methods
    public: 
        Network(int *layer_sizes, int N) : num_layers(N), sizes(layer_sizes) {
            
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<float> initial_value(0,1);

            biases.reserve(num_layers - 1);
            for(int i=1; i<num_layers; i++){
                std::cout << "Layer " << i << " Biases:\n";
                biases[i-1].reserve(sizes[i]);
                for(int j=0; j<sizes[i]; j++){
                    biases[i-1][j] = initial_value(generator);
                    std::cout << "    " << biases[i-1][j] << std::endl;
                }
            }
        }
};

#endif
