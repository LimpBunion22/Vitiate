#ifndef PYTHONHANDLER
#define PYTHONHANDLER

#include "network.h"
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

using namespace std;
namespace py = pybind11;

#define N_INPUTS "n_inputs"
#define N_LAYERS "n_layers"
#define NN_PER_LAYER "nnpl"

template <class T>
network<T> load_network(std::string network_name, bool derivate)
{
    py::scoped_interpreter guard{};
    py::module_ json = py::module_::import("json_handler");
    // py::module_ csv = py::module_::import("csv_handler");

    json.attr("load_json")(network_name);
    size_t n_inputs = json.attr("read_main_param")(N_INPUTS).cast<size_t>();
    size_t n_layers = json.attr("read_main_param")(N_LAYERS).cast<size_t>();
    std::vector<size_t> _nnpl;
    _nnpl.reserve(n_layers);

    for (int l = 0; l < n_layers; l++)
        _nnpl.emplace_back(json.attr("read_vector_param")(NN_PER_LAYER, l).cast<size_t>());

    myVec<size_t> nnpl(_nnpl.size(), &_nnpl);
    network<T> my_network(n_inputs, nnpl, derivate);

    return my_network;
}

template <class T>
void save_network(std::string network_name, network<T> my_network)
{
    return;
}

#endif
