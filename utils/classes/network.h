#ifndef NETWORK_H
#define NETWORK_H
#include <mathStructs.h>
#include <stdio.h>

using namespace std;

template <class T>
class network
{
private:
    size_t n_ins;
    size_t n_outs;
    size_t n_layers;
    vector<myMatrix<T>> params;
    vector<myVec<myFun<T>>> activations;
    vector<myVec<T>> inner_vals;
    vector<myVec<T>> bias;
    vector<myVec<T>> fx_activations;
    myVec<size_t> neurons_per_layer;
    vector<myMatrix<T>> fx_params;
    vector<myVec<T>> fx_bias;
    vector<myVec<T>> tmp_gradient;

public:
    network(size_t n_ins, myVec<size_t> &neurons_per_layer, bool derivate) : n_ins(n_ins),
                                                                             n_layers(neurons_per_layer.size),
                                                                             n_outs(neurons_per_layer[neurons_per_layer.size - 1]),
                                                                             neurons_per_layer(neurons_per_layer)
    {
        params.reserve(n_layers);
        activations.reserve(n_layers);
        inner_vals.reserve(n_layers);
        bias.reserve(n_layers);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
                params.emplace_back(neurons_per_layer[i], n_ins, RANDOM);
            else
                params.emplace_back(neurons_per_layer[i], neurons_per_layer[i - 1], RANDOM);

            activations.emplace_back(neurons_per_layer[i], derivate, RANDOM);
            inner_vals.emplace_back(neurons_per_layer[i], RANDOM);
            bias.emplace_back(neurons_per_layer[i], RANDOM);
        }
    }

    network(size_t n_ins, myVec<size_t> &neurons_per_layer, bool derivate,
            vector<vector<vector<T>>> *p, vector<vector<T>> *b) : n_ins(n_ins),
                                                                  n_layers(neurons_per_layer.size),
                                                                  n_outs(neurons_per_layer[neurons_per_layer.size - 1]),
                                                                  neurons_per_layer(neurons_per_layer)
    {
        params.reserve(n_layers);
        activations.reserve(n_layers);
        inner_vals.reserve(n_layers);
        bias.reserve(n_layers);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
                params.emplace_back(neurons_per_layer[i], n_ins, &(*p)[i]);
            else
                params.emplace_back(neurons_per_layer[i], neurons_per_layer[i - 1], &(*p)[i]);

            activations.emplace_back(neurons_per_layer[i], derivate, RANDOM);
            inner_vals.emplace_back(neurons_per_layer[i], RANDOM);
            bias.emplace_back(neurons_per_layer[i], &(*b)[i]);
        }
    }

    network &operator=(const network &rh)
    {
        if (this != &rh)
        {
            n_ins = rh.n_ins;
            n_outs = rh.n_outs;
            n_layers = rh.n_layers;
            params = rh.params;
            activations = rh.activations;
            inner_vals = rh.inner_vals;
            bias = rh.bias;
            fx_activations = rh.fx_activations;
            neurons_per_layer = rh.neurons_per_layer;
            fx_params = rh.fx_params;
            fx_bias = rh.bias;
            tmp_gradient = rh.tmp_gradient;
        }

        return *this;
    }

    network &operator=(network &&rh)
    {
        if (this != &rh)
        {
            n_ins = rh.n_ins;
            n_outs = rh.n_outs;
            n_layers = rh.n_layers;
            params = move(rh.params);
            activations = move(rh.activations);
            inner_vals = move(rh.inner_vals);
            bias = move(rh.bias);
            fx_activations = move(rh.fx_activations);
            neurons_per_layer = move(rh.neurons_per_layer);
            fx_params = move(rh.fx_params);
            fx_bias = move(rh.bias);
            tmp_gradient = move(rh.tmp_gradient);
        }

        return *this;
    }

    network(const network &rh) : n_ins(rh.n_ins),
                                 n_outs(rh.n_outs),
                                 n_layers(rh.n_layers),
                                 params(rh.params),
                                 activations(rh.activations),
                                 inner_vals(rh.inner_vals),
                                 bias(rh.bias),
                                 fx_activations(rh.fx_activations),
                                 neurons_per_layer(rh.neurons_per_layer),
                                 fx_params(rh.fx_params),
                                 fx_bias(rh.bias),
                                 tmp_gradient(rh.tmp_gradient)
    {
    }

    network(network &&rh) : n_ins(rh.n_ins),
                            n_outs(rh.n_outs),
                            n_layers(rh.n_layers),
                            params(move(rh.params)),
                            activations(move(rh.activations)),
                            inner_vals(move(rh.inner_vals)),
                            bias(move(rh.bias)),
                            fx_activations(move(rh.fx_activations)),
                            neurons_per_layer(move(rh.neurons_per_layer)),
                            fx_params(move(rh.fx_params)),
                            fx_bias(move(rh.bias)),
                            tmp_gradient(move(rh.tmp_gradient))
    {
    }

    void forward(myVec<T> &ins)
    {
        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                myVec<T> x = params[i] * ins + bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
            else
            {
                myVec<T> x = params[i] * inner_vals[i - 1] + bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
        }
    }

    void forwardGradient(myVec<T> &ins)
    {
        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                myVec<T> x = params[i] * ins + bias[i];
                inner_vals[i] = activations[i].calculate(x);
                fx_activations[i] = activations[i].derivate(x);
            }
            else
            {
                myVec<T> x = params[i] * inner_vals[i - 1] + bias[i];
                inner_vals[i] = activations[i].calculate(x);
                fx_activations[i] = activations[i].derivate(x);
            }
        }
    }

    void initGradient()
    {
        fx_activations.reserve(n_layers);
        fx_params.reserve(n_layers);
        fx_bias.reserve(n_layers);
        tmp_gradient.reserve(n_layers);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
                fx_params.emplace_back(neurons_per_layer[i], n_ins, RANDOM);
            else
                fx_params.emplace_back(neurons_per_layer[i], neurons_per_layer[i - 1], RANDOM);

            fx_activations.emplace_back(neurons_per_layer[i], RANDOM);
            fx_bias.emplace_back(neurons_per_layer[i], RANDOM);
            tmp_gradient.emplace_back(neurons_per_layer[i], RANDOM);
        }
    }

    //* R*(fx^P)*(fx^P)...^(fx matrix I)
    void gradient(myVec<T> &ins, myVec<T> set_outs)
    {
        forwardGradient(ins);
        myVec<T> R = set_outs - inner_vals[n_layers - 1];

        //* caso salidas
        tmp_gradient[n_layers - 1] = R;
        fx_params[n_layers - 1] = makeFrom(fx_activations[n_layers - 1], inner_vals[n_layers - 2]) ^ tmp_gradient[n_layers - 1];
        fx_bias[n_layers - 1] = tmp_gradient[n_layers - 1] ^ fx_activations[n_layers - 1];

        //* caso general
        for (int i = n_layers - 2; i > 0; i--)
        {
            myMatrix<T> fx_special_product_params = params[i + 1] ^ fx_activations[i + 1];
            tmp_gradient[i] = tmp_gradient[i + 1] * fx_special_product_params;
            fx_params[i] = makeFrom(fx_activations[i], inner_vals[i - 1]) ^ tmp_gradient[i];
            fx_bias[i] = tmp_gradient[i] ^ fx_activations[i];
        }

        //* caso entradas
        myMatrix<T> fx_special_product_params = params[1] ^ fx_activations[1];
        tmp_gradient[0] = tmp_gradient[1] * fx_special_product_params;
        fx_params[0] = makeFrom(fx_activations[0], ins) ^ tmp_gradient[0];
        fx_bias[0] = tmp_gradient[0] ^ fx_activations[0];
    }

    void printParams()
    {
        for (int i = 0; i < n_layers; i++)
        {
            cout << "Parámetros capa " << i << endl
                 << endl;
            params[i].print();
            cout << "Bias " << endl
                 << endl;
            bias[i].print();
            cout << endl;
        }
    }

    void printInnerVals()
    {
        cout << "Valores internos" << endl
             << endl;
        for (auto &i : inner_vals)
            i.print();
        cout << endl;
    }

    void printfxActivations()
    {

        cout << "fx activations" << endl
             << endl;
        for (auto &i : fx_activations)
            i.print();
        cout << endl;
    }

    void printfxNet()
    {
        for (int i = 0; i < n_layers; i++)
        {
            cout << "Gradiente parámetros capa " << i << endl
                 << endl;
            fx_params[i].print();
            cout << "Gradiente Bias " << endl
                 << endl;
            fx_bias[i].print();
            cout << endl;
        }
    }

    size_t get_n_ins() const
    {
        return n_ins;
    }

    size_t get_n_layers() const
    {
        return n_layers;
    }

    size_t get_neurons_per_layer(size_t i)
    {
        return neurons_per_layer[i];
    }

    T get_params(size_t i, size_t j, size_t k)
    {
        if (i < params.size())
            return params[i][j][k];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }

    T get_bias(size_t i, size_t j)

    {
        if (i < bias.size())
            return bias[i][j];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
    }
};
#endif