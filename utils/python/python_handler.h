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
    py::module_ csv = py::module_::import("csv_handler");

    json.attr("load_json")(network_name);
    size_t n_inputs = json.attr("read_main_param")(N_INPUTS).cast<size_t>();
    size_t n_layers = json.attr("read_main_param")(N_LAYERS).cast<size_t>();
    std::vector<size_t> _nnpl;
    _nnpl.reserve(n_layers);

    for (int l = 0; l < n_layers; l++)
        _nnpl.emplace_back(json.attr("read_vector_param")(NN_PER_LAYER, l).cast<size_t>());

    myVec<size_t> nnpl(_nnpl.size(), &_nnpl);
    vector<vector<vector<T>>> params(n_layers);
    vector<vector<T>> bias(n_layers);
    csv.attr("load_csv")(network_name);

    int row_count = 0;

    for (int i = 0; i < n_layers; i++)
    {
        params[i].reserve(nnpl[i]);
        vector<T> v_bias(nnpl[i]);

        for (int j = 0; j < nnpl[i]; j++)
        {
            int n_params;

            if (i == 0)
                n_params = n_inputs;
            else
                n_params = nnpl[i - 1];

            vector<T> v_params(n_params);

            for (int k = 0; k < n_params; k++)
                v_params[k] = csv.attr("read_value")(row_count, k).cast<T>();

            v_bias[j] = csv.attr("read_value")(row_count, n_params).cast<T>();
            params[i].emplace_back(v_params);
            row_count++;
        }

        bias[i] = v_bias;
    }

    return network<T>(n_inputs, nnpl, derivate, &params, &bias);
}

template <class T>
void save_network(std::string network_name, network<T> &my_network)
{
    py::scoped_interpreter guard{};
    py::module_ json = py::module_::import("json_handler");
    py::module_ csv = py::module_::import("csv_handler");

    json.attr("write_main_param")(N_INPUTS, my_network.get_n_ins());
    size_t n_layers = my_network.get_n_layers();
    json.attr("write_main_param")(N_LAYERS, n_layers);

    for (int l = 0; l < n_layers; l++)
        json.attr("write_vector_param")(NN_PER_LAYER, l, my_network.get_neurons_per_layer(l));

    json.attr("save_json")(network_name);

    int row_count = 0;

    for (int i = 0; i < n_layers; i++)
    {
        for (int j = 0; j < my_network.get_neurons_per_layer(i); j++)
        {
            int n_params;

            if (i == 0)
                n_params = my_network.get_n_ins();
            else
                n_params = my_network.get_neurons_per_layer(i - 1);

            vector<T> v_params(n_params);

            for (int k = 0; k < n_params; k++)
                csv.attr("write_value")(row_count, k, my_network.get_params(i, j, k));

            csv.attr("write_value")(row_count, n_params, my_network.get_bias(i, j));
            row_count++;
        }
    }

    csv.attr("save_csv")(network_name);

    return;
}

template <class T>
void save_xy(std::string plot_name, vector<T> &x_vector,  vector<T> &y_vector)
{
    py::scoped_interpreter guard{};
    py::module_ in_out = py::module_::import("in_out");

    std::string x_name = plot_name + "_x";
    std::string y_name = plot_name + "_y";
    in_out.attr("add_key")(x_name);
    in_out.attr("add_key")(y_name);

    for(int cnt= 0; cnt<x_vector.size(); cnt++){
        in_out.attr("add_value")(x_name, x_vector[cnt]);
        in_out.attr("add_value")(y_name, y_vector[cnt]);
    }

    in_out.attr("save_dict")(plot_name);

    return;
}

#endif
