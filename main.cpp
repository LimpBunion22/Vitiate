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

    myVec<float> ins = {2.0f, 1.0f};
    myVec<float> set_outs = {0, 0, 0, 0, 0};
    myVec<size_t> neurons_per_layer = {2, 3, 5};
    network<float> my_network(ins.size, neurons_per_layer, DERIVATE);
    // network<float> my_network(ins.size, neurons_per_layer, DERIVATE, &params, &bias);
    //network<float> my_network = load_network<float>("test_net", DERIVATE);
    network<float>::fx_container my_container(my_network);
    my_network.print_params();
    my_network.initGradient();

    int iterations = 10;
    vector<float> x_vector(iterations);
    vector<float> y_vector(iterations);

    for (int i = 0; i < iterations; i++)
    {
        my_network.gradient(ins, set_outs, my_container);
        // my_network.print_inner_vals();
        // my_network.print_fx_activations();
        // my_container.print_fx();
        my_container.normalize_1();
        // my_container.print_fx();
        my_network.update_params(my_container);

        int error = 0;
        myVec<float> net_outs = my_network.get_output();
        for(int h = 0; h<set_outs.size; h++)
            error += (set_outs[h] - net_outs[h])*(set_outs[h] - net_outs[h]);
        x_vector[i] = i;
        y_vector[i] = error;

    }
    save_xy("test_plot", x_vector,  y_vector);

    // my_network.print_params();
    my_network.print_inner_vals();
    //save_network<float>("saved_net", my_network);

    return 0;
}
