#include <pybind11/pybind11.h>
#include <stdio.h>
#include <network.h>
#include <time.h>
#include <pybind11/embed.h>
#include <python_handler.h>

namespace py = pybind11;
using namespace std;

int main()
{   
    srand(time(NULL));

    myVec<size_t> neurons_per_layer = {2, 3, 2};
    myVec<float> ins = {2.0f, 1.0f};
    myVec<float> set_outs = {0, 0};

    network<float> net(ins.size, neurons_per_layer, DERIVATE);
    net.printParams();
    net.initGradient();
    net.gradient(ins, set_outs);
    net.printInnerVals();
    net.printfxActivations();
    net.printfxNet();

    network<float> my_network = load_network<float>("test_json",true);
	return 0;
}

