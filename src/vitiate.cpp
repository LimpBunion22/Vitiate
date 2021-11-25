// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <network.h>

// namespace pybi = pybind11;

// PYBIND11_MODULE(vitiate, m)
// {
//     m.doc() = "Vitiate AI library for floats";

//     pybi::class_<network<float>> net(m, "netfloat");

// }
#include <stdio.h>
#include <network.h>
#include <vector>

using namespace std;

int main()
{
    srand(time(NULL));

    size_t n_ins = 2;
    vector<vector<float>> set_ins = {{2.0f, 1.0f}};
    vector<vector<float>> set_outs = {{0, 0, 0, 0, 0}};
    vector<size_t> neurons_per_layer = {3, 4, 5};

    // vector<vector<vector<float>>> params =
    //     {{{1, 2},
    //       {-0.1, 0.1},
    //       {0, 0.1}},
    //      {{1, 2, -0.1},
    //       {1, -0.1, -0.1},
    //       {1, 2, -0.1},
    //       {1, 0.1, 0.1}},
    //      {{1, 2, -0.1, 0.1},
    //       {1, 2, -0.1, 0.1},
    //       {0.1, 2, -1, 0.1},
    //       {-0.1, 2, -1, 0.1},
    //       {2, 2, 0.7, 0.1}}};

    // vector<vector<float>> bias =
    //     {{-0.1, 0, -0.1},
    //      {0.1, 0.1, 0.1, 0.1},
    //      {0, 0, 0, 0, 0}};

    network<float> my_network(n_ins, neurons_per_layer, DERIVATE);
    my_network.init_gradient(set_ins, set_outs);
    my_network.launch_gradient(10);
    my_network.print_inner_vals();
    vector<float> test_ins = {2.0f, 1.0f};
    cout << "launch forward with 2, 1" << endl;
    my_network.launch_forward(test_ins);
    my_network.print_inner_vals();

    return 0;
}
