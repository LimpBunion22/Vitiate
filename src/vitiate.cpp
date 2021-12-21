#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <defines.h>
#include <netHandler.h>

namespace py = pybind11;
using namespace std;
PYBIND11_MAKE_OPAQUE(vector<DATA_TYPE>);

PYBIND11_MODULE(vitiate, m)
{
    m.doc() = "Vitiate AI library for floats";

    py::bind_vector<vector<DATA_TYPE>>(m, "v_data_type");
    m.attr("CPU") = py::size_t(net::CPU);
    m.attr("CUDA") = py::size_t(net::CUDA);
    m.attr("FPGA") = py::size_t(net::FPGA);
    m.attr("MULTI") = py::size_t(net::MULTI);
    m.attr("DERIVATE") = py::bool_(net::DERIVATE);
    m.attr("NOT_DERIVATE") = py::bool_(net::NOT_DERIVATE);
    m.attr("RANDOM") = py::bool_(net::RANDOM);
    m.attr("NOT_RANDOM") = py::bool_(net::NOT_RANDOM);
    m.attr("RELOAD_FILE") = py::bool_(net::RELOAD_FILE);
    m.attr("REUSE_FILE") = py::bool_(net::REUSE_FILE);

    py::class_<net::net_handler>(m, "net_handler")
        .def(py::init<const string &>(), py::arg("path"))
        .def("set_active_net", &net::net_handler::set_active_net, py::arg("net_key"))
        .def("net_create", &net::net_handler::net_create, py::arg("net_key"), py::arg("implementation"), py::arg("derivate"), py::arg("random"), py::arg("file"), py::arg("file_reload") = false)
        .def("active_net_launch_forward", &net::net_handler::active_net_launch_forward, py::arg("inputs"))
        .def("active_net_init_gradient", &net::net_handler::active_net_init_gradient, py::arg("file"), py::arg("file_reload") = false)
        .def("active_net_launch_gradient", &net::net_handler::active_net_launch_gradient, py::arg("iterations"))
        .def("active_net_print_inner_vals", &net::net_handler::active_net_print_inner_vals)
        .def("active_net_get_gradient_performance", &net::net_handler::active_net_get_gradient_performance)
        .def("active_net_get_forward_performance", &net::net_handler::active_net_get_forward_performance)
        .def("active_net_write_net_to_file", &net::net_handler::active_net_write_net_to_file, py::arg("file"));
}

// #include <netHandler.h>
// #include <iostream>

// int main()
// {
//     using namespace std;

//     net::net_handler handler("/home/gabi/workspace_development");
//     vector<DATA_TYPE> ins = {1, 2};

//     handler.net_create("cpu_float", net::CPU, net::DERIVATE, net::RANDOM, "net");
//     handler.set_active_net("cpu_float");
//     handler.active_net_init_gradient("sets");

//     vector<DATA_TYPE> errors = handler.active_net_launch_gradient(45);
//     cout << "gradient errors\n";
//     for (auto &i : errors)
//         cout << i << " ";
//     cout << "\n";
//     cout << "gradient performance was " << handler.active_net_get_gradient_performance() << " us\n";

//     vector<DATA_TYPE> outs = handler.active_net_launch_forward(ins);
//     cout << "forward outs\n";
//     for (auto &i : outs)
//         cout << i << " ";
//     cout << "\n";
//     cout << "foward performance was " << handler.active_net_get_forward_performance() << " us\n";
// }