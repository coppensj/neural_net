// 'data_types.h' SRC: J.Coppens 2020

#ifndef NEURAL_NET_DATATYPES_H_
#define NEURAL_NET_DATATYPES_H_

#include <Eigen/Core>

struct training_image {
    Eigen::MatrixXf pixels;
    Eigen::MatrixXf value = Eigen::MatrixXf::Zero(10,1);
};

struct image {
    Eigen::MatrixXf pixels;
    int value = 0;
};

#endif  // NEURAL_NET_DATATYPES_H_
