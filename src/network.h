// 'network.h' SRC: J.Coppens 2020

#ifndef NETWORK
#define NETWORK

#include <random>
#include <cmath>
#include <Eigen/Core>
#include <algorithm>    // std::shuffle
#include "data_types.h" // training_image and image types

#include <iostream>

Eigen::MatrixXf sigmoid(const Eigen::MatrixXf& z){
    Eigen::MatrixXf z_new = z;
    for(int col=0; col<z.cols(); col++)
        for(int row=0; row<z.rows(); row++)    
            z_new(row,col) = 1.0/(1.0 + exp(-z(row,col)));
    return z_new;
}

Eigen::MatrixXf sigmoid_prime(const Eigen::MatrixXf& z){
    Eigen::MatrixXf z_new = sigmoid(z);
    for(int col=0; col<z.cols(); col++)
        for(int row=0; row<z.rows(); row++)    
            z_new(row,col) = z_new(row,col) * (1 - z_new(row,col));
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

            /* Eigen::MatrixXf a, test; */
            /* a.resize(2,1); */
            /* a << 1, 1; */
            /* test = feedforward(a); */
            /* std::cout << "test = \n" << test << std::endl; */
            /* std::cout << "test = \n" << sigmoid_prime(test) << std::endl; */
        }

        // Return the output of the network given input 'a'
        Eigen::MatrixXf feedforward(Eigen::MatrixXf& a){
            for(unsigned int i=0; i<biases.size(); i++)
                a = sigmoid(weights[i] * a + biases[i]);
            return a;
        }

        // train the neural-network using mini-batch stochastic gradient descent
        void SGD(std::vector<training_image> training_data, int epochs, int mini_batch_size, float eta, std::vector<image> test_data={})
        {
            std::random_device rd;
            std::mt19937 rng(rd());
            
            for(int i=0; i<epochs; i++)
            {
                // shuffle training data
                std::shuffle(training_data.begin(), training_data.end(), rng);
                
                for(int x=0; x<10; x++)
                    std::cout << "training[x] = " << training_data[x].value.transpose() << std::endl;
                //create mini_batches
                int n_mini_batches = 0;
                n_mini_batches = training_data.size() / mini_batch_size;
                std::cout << n_mini_batches << std::endl;
                /* for(int j=0; j<n_mini_batches; j++) */ 
                    //update_mini_batch(mini_batches[j]);
               
                if(!test_data.empty())
                    /* printf("Epoch %d: %f / %d\n", i, evaluate(test_data), n_test_images); */
                    printf("Epoch %d: %f / %d\n", i, -999.0, int(test_data.size()));
                else
                    printf("Epoch %d complete.\n", i);
            }
        }

    private:
        void update_mini_batch(/*mini_batch, float eta*/){
        }

        void evaluate(/*test_data*/){
        }
};

#endif
