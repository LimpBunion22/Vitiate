#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <network.h>

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(vector<size_t>);
PYBIND11_MAKE_OPAQUE(vector<float>);
PYBIND11_MAKE_OPAQUE(vector<vector<float>>);

static void rand_init(bool repeatable)
{
    static bool already_init = false;

    if (!already_init)
    {
        if (repeatable)
            srand(1);
        else
            srand(time(NULL));

        already_init = true;
    }
    else
        cout << "rand already init!\n";
}

PYBIND11_MODULE(vitiate, m)
{
    m.doc() = "Vitiate AI library for floats";

    m.def("rand_init", &rand_init, py::arg("repeteable"));
    py::bind_vector<vector<size_t>>(m, "net_structure");
    py::bind_vector<vector<float>>(m, "set_single");
    py::bind_vector<vector<vector<float>>>(m, "set");

    py::class_<network<float>>(m, "net_float")
        .def(py::init<size_t, vector<size_t> &, bool>(), py::arg("n_ins"), py::arg("n_p_l"), py::arg("derivate"))
        //.def(py::init<size_t, vector<vector<vector<float>>> &, vector<vector<float>> &, bool>())
        .def("launch_forward", &network<float>::launch_forward, py::arg("intpus"))
        .def("init_gradient", &network<float>::init_gradient, py::arg("set_ins"), py::arg("set_outs"))
        .def("launch_gradient", &network<float>::launch_gradient, py::arg("iterations"))
        .def("print_inner_vals", &network<float>::print_inner_vals)
        .def("get_gradient_performance", &network<float>::get_gradient_performance)
        .def("get_forward_performance", &network<float>::get_forward_performance);
}
// #include <stdio.h>
// #include <network.h>
// #include <vector>

// using namespace std;

// int main()
// {
//     srand(time(NULL));

//     size_t n_ins = 2;
//     vector<vector<float>> set_ins = {{2.0f, 1.0f}};
//     vector<vector<float>> set_outs = {{0, 0, 0, 0, 0}};
//     vector<size_t> neurons_per_layer = {3, 4, 5};

//     // vector<vector<vector<float>>> params =
//     //     {{{1, 2},
//     //       {-0.1, 0.1},
//     //       {0, 0.1}},
//     //      {{1, 2, -0.1},
//     //       {1, -0.1, -0.1},
//     //       {1, 2, -0.1},
//     //       {1, 0.1, 0.1}},
//     //      {{1, 2, -0.1, 0.1},
//     //       {1, 2, -0.1, 0.1},
//     //       {0.1, 2, -1, 0.1},
//     //       {-0.1, 2, -1, 0.1},
//     //       {2, 2, 0.7, 0.1}}};

//     // vector<vector<float>> bias =
//     //     {{-0.1, 0, -0.1},
//     //      {0.1, 0.1, 0.1, 0.1},
//     //      {0, 0, 0, 0, 0}};

//     network<float> my_network(n_ins, neurons_per_layer, DERIVATE);
//     my_network.init_gradient(set_ins, set_outs);
//     my_network.launch_gradient(10);
//     my_network.print_inner_vals();
//     vector<float> test_ins = {2.0f, 1.0f};
//     cout << "launch forward with 2, 1" << endl;
//     my_network.launch_forward(test_ins);
//     my_network.print_inner_vals();

//     return 0;
// }
