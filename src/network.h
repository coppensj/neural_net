// 'network.h' SRC: J.Coppens 2020

#ifndef NETWORK
#define NETWORK

#include <random>
#include <cmath>
#include <Eigen/Core>

#include <iostream>

std::vector<float> sigmoid(std::vector<float> &z){
    std::vector<float> z_new(z.size());
    for(unsigned int i=0; i<z.size(); i++)
        z_new[i] = 1.0/(1.0 + exp(-z[i]));
    return z_new;
}

std::vector<std::vector<float>> sigmoid(std::vector<std::vector<float>> &z){
    std::vector<std::vector<float>> z_new(z.size());
    for(unsigned int i=0; i<z.size(); i++){
        z_new[i].resize(z[i].size());
        for(unsigned int j=0; j<z[i].size(); j++){
            z_new[i][j] = 1.0/(1.0 + exp(-z[i][j]));
        }
    }
    return z_new;
}

class Network {
    // Data
    public:
        int num_layers;
        int *sizes;
        std::vector<std::vector<float>> biases;
        std::vector<std::vector<std::vector<float>>> weights;
        Eigen::Matrix<float, 3, 1> t;

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
                
                std::cout << "weight " << i << "(" << sizes[i+1] << "," << sizes[i] << ")\n";
                for(int row=0; row<sizes[i+1]; row++){    
                    weights[i][row].resize(sizes[i]);
                    for(int col=0; col<sizes[i]; col++){
                        weights[i][row][col] = initial_value(generator);
                        std::cout << weights[i][row][col] << " "; //for testing
                    }
                    std::cout << std::endl; //for testing
                }
                std::cout << std::endl; //for testing
            }
            
            /* std::vector<float> test = sigmoid(biases[0]); */
            /* for(unsigned int i=0; i<test.size(); i++) */
            /*     std::cout << biases[0][i] << " " << test[i] << std::endl; */

            /* std::vector<std::vector<float>> test2d = sigmoid(weights[0]); */
            /* for(unsigned int i=0; i<test2d.size(); i++) */
            /*     for(unsigned int j=0; j<test2d[i].size(); j++) */
            /*         std::cout << i << " " << j << " " << weights[0][i][j] << " " << test2d[i][j] << std::endl; */
        }
};

#endif
