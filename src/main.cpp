// 'main.cpp' SRC: J.Coppens 2020
// main program for testing neural net 

#include "network.h"

int main(){

    int layer_sizes[4] = {2, 6, 4, 2};

    Network net(layer_sizes, 4);

    return 0;
}
