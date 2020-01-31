// 'network.cpp' SRC: J.Coppens 2020

#ifndef NETWORK
#define NETWORK

#include <random>

class Network {
    // Data
    public:
        int num_layers;
        int *sizes;
        float *biases;
        float *weights;

    // Methods
    public: 
        Network(int *layer_sizes, int N) : num_layers(N), sizes(layer_sizes) {
            
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<float> inital_value(0,1);
        }

};

#endif
