// 'network.h' SRC: J.Coppens 2020

#ifndef NEURAL_NET_NETWORK_H_
#define NEURAL_NET_NETWORK_H_

#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Core>
#include <algorithm>    // std::shuffle
#include <vector>       // std::vector

#include "data_types.h" // training_image and image types

Eigen::MatrixXf 
Sigmoid(const Eigen::MatrixXf& z) {
    Eigen::MatrixXf z_new = z;
    for (int col=0; col<z.cols(); col++)
        for (int row=0; row<z.rows(); row++)    
            z_new(row,col) = 1.0/(1.0 + exp(-z(row,col)));
    return z_new;
}

Eigen::MatrixXf 
SigmoidPrime(const Eigen::MatrixXf& z) {
    Eigen::MatrixXf z_new = Sigmoid(z);
    for (int col=0; col<z.cols(); col++)
        for (int row=0; row<z.rows(); row++)    
            z_new(row,col) = z_new(row,col) * (1 - z_new(row,col));
    return z_new;
}

class NeuralNetwork {
    // Data
    public:
        int num_layers_;
        int *layer_sizes_;
        std::vector<Eigen::MatrixXf> biases_;
        std::vector<Eigen::MatrixXf> weights_;

    // Methods
    public: 
        NeuralNetwork(int *layer_sizes, int num_layers) 
            : num_layers_(num_layers), layer_sizes_(layer_sizes) {
            
            std::random_device rd;
            std::mt19937 generator(rd());
            std::normal_distribution<float> initial_value(0,1);

            // Initialize biases
            biases_.resize(num_layers_ - 1);
            for (int i=0; i<biases_.size(); i++) {
                biases_[i].resize(layer_sizes_[i+1],1);
                for(int row=0; row<biases_[i].rows(); row++)
                    biases_[i](row, 0) = initial_value(generator);
            }
           
            // Initialize weights
            weights_.resize(num_layers_ - 1);
            for (int i=0; i<weights_.size(); i++) {
                weights_[i].resize(layer_sizes_[i+1], layer_sizes_[i]);
                for (int col=0; col<weights_[i].cols(); col++)
                    for (int row=0; row<weights_[i].rows(); row++)    
                        weights_[i](row,col) = initial_value(generator);
            }

            /* Eigen::MatrixXf a, test; */
            /* a.resize(2,1); */
            /* a << 1, 1; */
            /* test = feedforward(a); */
            /* std::cout << "test = " << test << std::endl; */
            /* std::cout << "test = " << SigmoidPrime(test) << std::endl; */
        }

        // Return the output of the network given input 'a'
        Eigen::MatrixXf feedforward(Eigen::MatrixXf& a) {
            for (unsigned int i=0; i<biases_.size(); i++)
                a = Sigmoid(weights_[i] * a + biases_[i]);
            return a;
        }

        // train the neural-network using mini-batch stochastic gradient descent
        void SGD(std::vector<training_image> training_data, int epochs, int mini_batch_size, 
                float eta, std::vector<image> test_data={}) {
            std::random_device rd;
            std::mt19937 rng(rd());
            
            for (int i=0; i<epochs; i++) {
                // shuffle training data
                std::shuffle(training_data.begin(), training_data.end(), rng);
                
                //create and update mini_batches
                int n_mini_batches = (training_data.size() + mini_batch_size - 1) / mini_batch_size;
                for (int j=0; j<n_mini_batches; j++) {
                    UpdateMiniBatch(&training_data[0], j * mini_batch_size, 
                            std::min( (j + 1) * mini_batch_size, int(training_data.size()) ), eta);
                }

                if (test_data.empty())
                    printf("Epoch %d complete.\n", i);
                else
                    printf("Epoch %d: %d / %d\n", i, Evaluate(test_data), int(test_data.size()));
                
                break; /////<<---- REMOVE LATER
            }
        }

    private:
        void UpdateMiniBatch(training_image *mini_batch, int start, int end, float eta) {
            int mini_batch_len = end - start;

            std::vector<Eigen::MatrixXf> nabla_b;
            nabla_b.resize(biases_.size());
            for (int i=0; i<nabla_b.size(); i++)
                nabla_b[i] = Eigen::MatrixXf::Zero(biases_[i].rows(), biases_[i].cols());
            
            std::vector<Eigen::MatrixXf> nabla_w;
            nabla_w.resize(weights_.size());
            for (int i=0; i<nabla_w.size(); i++)
                nabla_w[i] = Eigen::MatrixXf::Zero(weights_[i].rows(), weights_[i].cols());

            /* for (int i=start; i<end; i++) { */
            /*     std::cout << i << " " << mini_batch[i].value.transpose() << std::endl; */
            /*     /1* auto [delta_nabla_b, delta_nabla_w] = backprop(mini_batch[i].pixels, mini_batch[i].value); *1/ */
                
            /*     /1* for(int j=0; j<nabla_b.size(); j++) *1/ */
            /*     /1*     nabla_b[j] += delta_nabla_b[j]; *1/ */
            /*     /1* for(int j=0; j<nabla_w.size(); j++) *1/ */
            /*     /1*     nabla_w[j] += delta_nabla_w[j]; *1/ */
                
            /*     /1* for(int j=0; j<biases_.size(); j++) *1/ */
            /*     /1*     biases_[i]  = biases_ - (eta / mini_batch_len) * nabla_b[i]; *1/ */
            /*     /1* for(int j=0; j<weights_.size(); j++) *1/ */
            /*     /1*     weights_[i] = weights_ - (eta / mini_batch_len) * nabla_w[i]; *1/ */
            /* } */
        }

        int Evaluate(std::vector<image> test_data) {
            return -1;
        }
};  // class NeuralNetwork

#endif  // NEURAL_NET_NETWORK_H_

