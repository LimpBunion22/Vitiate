#include "python_handler.h"
#include "network.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>



namespace py = pybind11;

template <class T>
network <T> load_network(std::string network_name, bool derivate){
    py::scoped_interpreter guard{};
    py::module_ json = py::module_::import("json_handler");
    py::module_ csv = py::module_::import("csv_handler");

    json.attr("load_json")(network_name);
    int n_inputs  = json.attr("read_main_param")(N_INPUTS).cast<int>();
    int n_layers  = json.attr("read_main_param")(N_LAYERS).cast<int>();
    std::vector nnpl = int[n_layers];
    for (l = 0; l<n_layers; l++)
        nnpl[l]  = json.attr("read_main_param")(NN_PER_LAYER,l).cast<int>();

    network <T> my_network(n_inputs, nnpl, derivate);

    return my_network;
}

template <class T>
void save_network(std::string network_name, network <T> my_network){
    return ;
}