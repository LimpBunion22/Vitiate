#ifndef NETWORK_H
#define NETWORK_H
#include <mathStructs.h>
#include <stdio.h>

using namespace std;

constexpr size_t BIAS = 1;

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
                params.emplace_back(neurons_per_layer[i], n_ins, nullptr);
            else
                params.emplace_back(neurons_per_layer[i], neurons_per_layer[i - 1], nullptr);

            activations.emplace_back(neurons_per_layer[i], derivate, nullptr);
            inner_vals.emplace_back(neurons_per_layer[i], nullptr);
            bias.emplace_back(neurons_per_layer[i], nullptr);
        }
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
                fx_params.emplace_back(neurons_per_layer[i], n_ins, nullptr);
            else
                fx_params.emplace_back(neurons_per_layer[i], neurons_per_layer[i - 1], nullptr);

            fx_activations.emplace_back(neurons_per_layer[i], nullptr);
            fx_bias.emplace_back(neurons_per_layer[i], nullptr);
            tmp_gradient.emplace_back(neurons_per_layer[i], nullptr);
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
};
#endif