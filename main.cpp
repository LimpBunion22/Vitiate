#include <stdio.h>
#include <network.h>
#include <time.h>
#include <python_handler.h>

namespace py = pybind11;
using namespace std;

int main()
{
    srand(time(NULL));

    // myVec<size_t> neurons_per_layer = {2, 3, 2};
    myVec<float> ins = {2.0f, 1.0f};
    myVec<float> set_outs = {0, 0, 0, 0 ,0};

    // network<float> net(ins.size, neurons_per_layer, DERIVATE);
    // net.printParams();
    // net.initGradient();
    // net.gradient(ins, set_outs);
    // net.printInnerVals();
    // net.printfxActivations();
    // net.printfxNet();

    // vector<vector<vector<float>>> params =
    //     {{{1, 3},
    //       {5, 7}},
    //      {{1, 1},
    //       {2, 1},
    //       {3, 2}},
    //      {{1, 1, 5},
    //       {2, 2, 3}}};

    // vector<vector<float>> bias =
    //     {{1, 5},
    //      {3, 4, 1},
    //      {2, 2}};

    // network<float> my_network(ins.size, neurons_per_layer, DERIVATE, &params, &bias);
    network<float> my_network = load_network<float>("test_net", DERIVATE);
    my_network.printParams();
    my_network.initGradient();
    my_network.gradient(ins, set_outs);
    my_network.printInnerVals();
    my_network.printfxActivations();
    my_network.printfxNet();

    save_network<float>("saved_net", my_network);

    return 0;
}
