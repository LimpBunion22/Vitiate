#ifndef PYTHONHANDLER
#define PYTHONHANDLER

#include "network.h"

using namespace std;

#define N_INPUTS "n_inputs"
#define N_LAYERS "n_layers"
#define NN_PER_LAYER "nnpl"

template <class T>
network <T>load_network(std::string network_name, bool derivate);

template <class T>
void save_network(std::string network_name, network <T> my_network);

#endif
