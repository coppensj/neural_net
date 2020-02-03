// 'network.h' SRC: J.Coppens 2020

#ifndef NETWORK
#define NETWORK

#include <random>
#include <cmath>

#include <iostream>

std::vector<float> sigmoid(std::vector<float> &z){
    std::vector<float> z_new(z.size());
    for(int i=0; i<z.size(); i++)
        z_new[i] = 1.0/(1.0 + exp(-z[i]));
    return z_new;
}

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

            // Initialize biases
            biases.resize(num_layers - 1);
            for(int i=0; i<num_layers-1; i++){
                biases[i].resize(sizes[i+1]);
                for(int j=0; j<sizes[i+1]; j++){
                    biases[i][j] = initial_value(generator);
                }
            }
           
            // Initialize weights
            weights.resize(num_layers - 1);
            for(int i=0; i<num_layers-1; i++){
                weights[i].resize(sizes[i+1]);
                
                for(int row=0; row<sizes[i+1]; row++){    
                    weights[i][row].resize(sizes[i]);
                    for(int col=0; col<sizes[i]; col++)
                        weights[i][row][col] = initial_value(generator);
                }
            }
            
            std::vector<float> test = sigmoid(biases[0]);
            for(int i=0; i<test.size(); i++)
                std::cout << biases[0][i] << " " << test[i] << std::endl;
        }
};

#endif
