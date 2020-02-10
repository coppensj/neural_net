// 'network.h' SRC: J.Coppens 2020

#ifndef NETWORK
#define NETWORK

#include <random>
#include <cmath>
#include <Eigen/Core>

#include <iostream>

Eigen::MatrixXf sigmoid(const Eigen::Ref<const Eigen::MatrixXf>& z){
    Eigen::MatrixXf z_new = z;
    for(int col=0; col<z.cols(); col++)
        for(int row=0; row<z.rows(); row++)    
            z_new(row,col) = 1.0/(1.0 + exp(-z(row,col)));
    return z_new;
}


class Network {
    // Data
    public:
        int num_layers;
        int *sizes;
        std::vector<Eigen::MatrixXf> biases;
        std::vector<Eigen::MatrixXf> weights;

    // Methods
    public: 
        Network(int *layer_sizes, int N) : num_layers(N), sizes(layer_sizes) {
            
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<float> initial_value(0,1);

            // Initialize biases
            biases.resize(num_layers - 1);
            for(int i=0; i<num_layers-1; i++){
                biases[i].resize(sizes[i+1],1);
                for(int j=0; j<sizes[i+1]; j++)
                    biases[i](j, 0) = initial_value(generator);
            }
           
            // Initialize weights
            weights.resize(num_layers - 1);
            for(int i=0; i<num_layers-1; i++){
                weights[i].resize(sizes[i+1], sizes[i]);
                for(int col=0; col<sizes[i]; col++)
                    for(int row=0; row<sizes[i+1]; row++)    
                        weights[i](row,col) = initial_value(generator);
            }

            Eigen::MatrixXf a, test;
            a.resize(2,1);
            a << 1, 1;
            test = feedforward(a);
            std::cout << "test = \n" << test << std::endl;
        }

        Eigen::MatrixXf feedforward(Eigen::MatrixXf& a){
            for(unsigned int i=0; i<biases.size(); i++)
                a = sigmoid(weights[i] * a + biases[i]);
            return a;
        }
};

#endif
