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
            for(int i=0; i<num_layers-1; i++){
                biases[i].reserve(sizes[i+1]);
                for(int j=0; j<sizes[i+1]; j++)
                    biases[i][j] = initial_value(generator);
            }
            
            weights.reserve(num_layers - 1);
            for(int i=1; i<num_layers; i++){
                weights[i-1].reserve(sizes[i]);
                
                for(int row=0; row<sizes[i]; row++){    
                    weights[i-1][row].reserve(sizes[i-1]);
                    
                    for(int col=0; col<sizes[i-1]; col++)
                        weights[i-1][row][col] = initial_value(generator);
                }
            }
            
            for(int i=0; i<num_layers-1; i++){
                weights[i].reserve(sizes[i+1]);
                
                for(int row=0; row<sizes[i+1]; row++){    
                    weights[i][row].reserve(sizes[i]);
                    
                    for(int col=0; col<sizes[i]; col++){
                        weights[i][row][col] = initial_value(generator);
                        std::cout << weights[i][row][col] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << std::endl;
            }
        
        }
};

#endif
