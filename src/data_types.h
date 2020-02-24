// 'data_types.h' SRC: J.Coppens 2020

#ifndef DATATYPES
#define DATATYPES

#include <Eigen/Core>

struct training_image {
    Eigen::MatrixXf pixels;
    Eigen::Matrix<float, 10, 1> value;

    training_image()
    {
        value << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
    }
};

struct image {
    Eigen::MatrixXf pixels;
    int value;

    image()
    {
        value = 0;
    }
};

#endif
