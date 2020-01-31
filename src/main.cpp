// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"

int main(){

    int layer_sizes[3] = {2, 3, 1};

    Network net(layer_sizes, 3);

    return 0;
}
