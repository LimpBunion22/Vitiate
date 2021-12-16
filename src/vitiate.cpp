// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>

// namespace py = pybind11;
// PYBIND11_MAKE_OPAQUE(vector<size_t>);
// PYBIND11_MAKE_OPAQUE(vector<float>);
// PYBIND11_MAKE_OPAQUE(vector<vector<float>>);

// static void rand_init(bool repeatable)
// {
//     static bool already_init = false;

//     if (!already_init)
//     {
//         if (repeatable)
//             srand(1);
//         else
//             srand(time(NULL));

//         already_init = true;
//     }
//     else
//         cout << "rand already init!\n";
// }

// PYBIND11_MODULE(vitiate, m)
// {
//     m.doc() = "Vitiate AI library for floats";

//     m.def("rand_init", &rand_init, py::arg("repeatable"));
//     py::bind_vector<vector<size_t>>(m, "v_size_t");
//     py::bind_vector<vector<float>>(m, "v_float");
//     py::bind_vector<vector<vector<float>>>(m, "vv_float");

//     py::class_<network<float>> net(m, "net_float");

//     py::class_<network<float>::file_manager>(net, "file_manager")
//         .def(py::init<>())
//         .def("load_net_structure", &network<float>::file_manager::load_net_structure)
//         .def("load_net", &network<float>::file_manager::load_net)
//         .def("load_sets", &network<float>::file_manager::load_sets)
//         .def("write_net_to_file", &network<float>::file_manager::write_net_to_file);

//     net.def(py::init<size_t, vector<size_t> &, bool>(), py::arg("n_ins"), py::arg("n_p_l"), py::arg("derivate"))
//         .def(py::init<size_t, vector<size_t> &, vector<vector<vector<float>>> &, vector<vector<float>> &, bool>())
//         .def(py::init<network<float>::file_manager &, bool, bool>(), py::arg("file_manager"), py::arg("derivate"), py::arg("random"))
//         .def("launch_forward", &network<float>::launch_forward, py::arg("inputs"))
//         .def("init_gradient", py::overload_cast<vector<vector<float>> &, vector<vector<float>> &>(&network<float>::init_gradient), py::arg("set_ins"), py::arg("set_outs"))
//         .def("init_gradient", py::overload_cast<network<float>::file_manager &>(&network<float>::init_gradient), py::arg("file_manager"))
//         .def("launch_gradient", &network<float>::launch_gradient, py::arg("iterations"))
//         .def("print_inner_vals", &network<float>::print_inner_vals)
//         .def("print_params", &network<float>::print_params)
//         .def("get_gradient_performance", &network<float>::get_gradient_performance)
//         .def("get_forward_performance", &network<float>::get_forward_performance);
// }

#include <netHandler.h>
#include <iostream>

int main()
{
    using namespace std;

    net::net_handler handler("/home/gabi");
    handler.net_create("cpu_float", net::CPU, net::DERIVATE, net::RANDOM, "net");
    handler.init_gradient("cpu_float", "sets");

    vector<DATA_TYPE> errors = handler.launch_gradient("cpu_float", 45);
    cout << "gradient errors\n";
    for (auto &i : errors)
        cout << i << " ";
    cout << "\n\n";
    cout << "gradient performance was " << handler.get_gradient_performance("cpu_float") << " us\n\n";

    vector<DATA_TYPE> ins = {1, 2};
    vector<DATA_TYPE> outs = handler.launch_forward("cpu_float", ins);
    cout << "forward outs\n";
    for (auto &i : outs)
        cout << i << " ";
    cout << "\n\n";
    cout << "foward performance was " << handler.get_forward_performance("cpu_float") << " us\n";
    
    handler.write_net_to_file("cpu_float", "cpu_float_values");
}