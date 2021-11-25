#ifndef NETWORK_H
#define NETWORK_H
#include <mathStructs.h>
#include <stdio.h>
#include <math.h>

using namespace std;

template <class T>
class network
{
private:
    class fx_container
    {
    private:
        size_t n_layers;

    public:
        vector<myMatrix<T>> fx_params;
        vector<myVec<T>> fx_bias;
        myVec<T> ins;
        myVec<T> outs;

    public:
        fx_container(vector<size_t> &n_p_l, size_t ins_num)
            : n_layers(n_p_l.size()), ins(0, CERO), outs(0, CERO)
        {
            fx_params.reserve(n_layers);
            fx_bias.reserve(n_layers);

            for (int i = 0; i < n_layers; i++)
            {
                if (i == 0)
                    fx_params.emplace_back(n_p_l[i], ins_num, CERO);
                else
                    fx_params.emplace_back(n_p_l[i], n_p_l[i - 1], CERO);

                fx_bias.emplace_back(n_p_l[i], CERO);
            }
        }

        fx_container(vector<size_t> &n_p_l, vector<T> &ins, vector<T> &outs)
            : n_layers(n_p_l.size()), ins(ins.size(), CONTENT, &ins), outs(outs.size(), CONTENT, &outs)
        {
            fx_params.reserve(n_layers);
            fx_bias.reserve(n_layers);

            for (int i = 0; i < n_layers; i++)
            {
                if (i == 0)
                    fx_params.emplace_back(n_p_l[i], this->ins.size(), CERO); //* puesto que ins.size() es 0, al mover su contenido
                else
                    fx_params.emplace_back(n_p_l[i], n_p_l[i - 1], CERO);

                fx_bias.emplace_back(n_p_l[i], CERO);
            }
        }

        fx_container(const fx_container &rh) : n_layers(rh.n_layers),
                                               fx_params(rh.fx_params),
                                               fx_bias(rh.fx_bias),
                                               ins(rh.ins),
                                               outs(rh.outs)
        {
        }

        fx_container(fx_container &&rh) : n_layers(rh.n_layers),
                                          fx_params(move(rh.fx_params)),
                                          fx_bias(move(rh.fx_bias)),
                                          ins(move(rh.ins)),
                                          outs(move(rh.outs))
        {
        }

        void reset()
        {
            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                        fx_params[i][j][k] = 0;

                    fx_bias[i][j] = 0;
                }
            }
        }

        fx_container &operator=(const fx_container &rh)
        {
            if (this != &rh)
            {
                n_layers = rh.n_layers;
                fx_params = rh.fx_params;
                fx_bias = rh.fx_bias;
                ins = rh.ins;
                outs = rh.outs;
            }

            return *this;
        }

        fx_container &operator+=(fx_container &rh)
        {
#ifdef ASSERT
            if (n_layers != rh.n_layers)
                cout << "invalid dimensions lh is " << n_layers << " rh is " << rh.n_layers << endl;
            else
#endif
                for (int i = 0; i < n_layers; i++)
                {
                    fx_params[i] += rh.fx_params[i];
                    fx_bias[i] += rh.fx_bias[i];
                }

            return *this;
        }

        fx_container &operator-=(fx_container &rh)
        {
#ifdef ASSERT
            if (n_layers != rh.n_layers)
                cout << "invalid dimensions lh is " << n_layers << " rh is " << rh.n_layers << endl;
            else
#endif
                for (int i = 0; i < n_layers; i++)
                {
                    fx_params[i] -= rh.fx_params[i];
                    fx_bias[i] -= rh.fx_bias[i];
                }

            return *this;
        }

        void normalize_0(T factor = 1)
        {
            T max = 0;
            T val = 0;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                    {
                        val = abs(fx_params[i][j][k]);
                        max = val > max ? val : max;
                    }

                    val = abs(fx_bias[i][j]);
                    max = val > max ? val : max;
                }
            }

            max *= factor;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                        fx_params[i][j][k] /= max;

                    fx_bias[i][j] /= max;
                }
            }
        }

        void normalize_1(T factor = 1)
        {
            T max = 0;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                        max += abs(fx_params[i][j][k]);

                    max += abs(fx_bias[i][j]);
                }
            }

            max *= factor;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                        fx_params[i][j][k] /= max;

                    fx_bias[i][j] /= max;
                }
            }
        }

        void normalize_2(T factor = 1)
        {
            T max = 0;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                        max += fx_params[i][j][k] * fx_params[i][j][k];

                    max += fx_bias[i][j] * fx_bias[i][j];
                }
            }

            max = sqrt(max);
            max *= factor;

            for (int i = 0; i < n_layers; i++)
            {
                for (int j = 0; j < fx_params[i].rows(); j++)
                {
                    for (int k = 0; k < fx_params[i].cols(); k++)
                        fx_params[i][j][k] /= max;

                    fx_bias[i][j] /= max;
                }
            }
        }

        void print_fx()
        {
            for (int i = 0; i < n_layers; i++)
            {
                cout << "Gradiente parámetros capa " << i << endl
                     << endl;
                fx_params[i].print();
                cout << "Gradiente Bias " << endl
                     << endl;
                fx_bias[i].print();
            }
        }
    };

private:
    size_t n_layers;
    vector<myMatrix<T>> params;
    vector<myVec<myFun<T>>> activations;
    vector<myVec<T>> inner_vals;
    vector<myVec<T>> bias;
    vector<myVec<T>> fx_activations;
    vector<myVec<T>> tmp_gradient;
    vector<fx_container> containers;
    size_t acum_pos;
    bool gradient_init;

private:
    void forward_gradient(myVec<T> &ins)
    {
        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                myVec<T> x = params[i] * ins;
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
                fx_activations[i] = activations[i].derivate(x);
            }
            else
            {
                myVec<T> x = params[i] * inner_vals[i - 1];
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
                fx_activations[i] = activations[i].derivate(x);
            }
        }
    }

    //* R*(fx^P)*(fx^P)...^(fx matrix I)
    void gradient(fx_container &fx)
    {
        forward_gradient(fx.ins);
        myVec<T> R = fx.outs - inner_vals[n_layers - 1];

        //* caso salidas
        tmp_gradient[n_layers - 1] = R;
        fx.fx_params[n_layers - 1] = make_from(fx_activations[n_layers - 1], inner_vals[n_layers - 2]);
        fx.fx_params[n_layers - 1] ^= tmp_gradient[n_layers - 1];
        fx.fx_bias[n_layers - 1] = tmp_gradient[n_layers - 1] ^ fx_activations[n_layers - 1];

        //* caso general
        for (int i = n_layers - 2; i > 0; i--)
        {
            myMatrix<T> fx_special_product_params = params[i + 1] ^ fx_activations[i + 1];
            tmp_gradient[i] = tmp_gradient[i + 1] * fx_special_product_params;
            fx.fx_params[i] = make_from(fx_activations[i], inner_vals[i - 1]);
            fx.fx_params[i] ^= tmp_gradient[i];
            fx.fx_bias[i] = tmp_gradient[i] ^ fx_activations[i];
        }

        //* caso entradas
        myMatrix<T> fx_special_product_params = params[1] ^ fx_activations[1];
        tmp_gradient[0] = tmp_gradient[1] * fx_special_product_params;
        fx.fx_params[0] = make_from(fx_activations[0], fx.ins);
        fx.fx_params[0] ^= tmp_gradient[0];
        fx.fx_bias[0] = tmp_gradient[0] ^ fx_activations[0];
    }

    void gradient_update_params(fx_container &fx)
    {
        for (int i = 0; i < n_layers; i++)
        {
            params[i] += fx.fx_params[i];
            bias[i] += fx.fx_bias[i];
        }
    }

public:
    network(size_t n_ins, vector<size_t> &n_p_l, bool derivate) : n_layers(n_p_l.size()), acum_pos(0), gradient_init(false)

    {
        params.reserve(n_layers);
        activations.reserve(n_layers);
        inner_vals.reserve(n_layers);
        bias.reserve(n_layers);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
                params.emplace_back(n_p_l[i], n_ins, RANDOM);
            else
                params.emplace_back(n_p_l[i], n_p_l[i - 1], RANDOM);

            activations.emplace_back(n_p_l[i], derivate);
            inner_vals.emplace_back(n_p_l[i], CERO);
            bias.emplace_back(n_p_l[i], RANDOM);
        }
    }

    network(size_t n_ins, vector<vector<vector<T>>> &p, vector<vector<T>> &b, bool derivate) : n_layers(b.size()), acum_pos(0), gradient_init(false)
    {
        vector<size_t> n_p_l;
        n_p_l.reserve(n_layers);
        params.reserve(n_layers);
        activations.reserve(n_layers);
        inner_vals.reserve(n_layers);
        bias.reserve(n_layers);

        for (int i = 0; i < n_layers; i++)
        {
            n_p_l.emplace_back(b[i].size()); //* para guardar el tamaño de b[i], puesto que mueves su contenido y el vector se hace 0

            if (i == 0)
                params.emplace_back(n_p_l[i], n_ins, CONTENT, &p[i]);
            else
                params.emplace_back(n_p_l[i], n_p_l[i - 1], CONTENT, &p[i]);

            activations.emplace_back(n_p_l[i], derivate);
            inner_vals.emplace_back(n_p_l[i], CERO);
            bias.emplace_back(n_p_l[i], CONTENT, &b[i]);
        }
    }

    network &operator=(const network &rh)
    {
        if (this != &rh)
        {
            n_layers = rh.n_layers;
            params = rh.params;
            activations = rh.activations;
            inner_vals = rh.inner_vals;
            bias = rh.bias;
            fx_activations = rh.fx_activations;
            tmp_gradient = rh.tmp_gradient;
            containers = rh.containers;
            acum_pos = rh.acum_pos;
            gradient_init = rh.gradient_init;
        }

        return *this;
    }

    network &operator=(network &&rh)
    {
        if (this != &rh)
        {
            n_layers = rh.n_layers;
            params = move(rh.params);
            activations = move(rh.activations);
            inner_vals = move(rh.inner_vals);
            bias = move(rh.bias);
            fx_activations = move(rh.fx_activations);
            tmp_gradient = move(rh.tmp_gradient);
            containers = move(rh.containers);
            acum_pos = rh.acum_pos;
            gradient_init = rh.gradient_init;
        }

        return *this;
    }

    network(const network &rh) : n_layers(rh.n_layers),
                                 params(rh.params),
                                 activations(rh.activations),
                                 inner_vals(rh.inner_vals),
                                 bias(rh.bias),
                                 fx_activations(rh.fx_activations),
                                 tmp_gradient(rh.tmp_gradient),
                                 containers(rh.containers),
                                 acum_pos(rh.acum_pos),
                                 gradient_init(rh.gradient_init)
    {
    }

    network(network &&rh) : n_layers(rh.n_layers),
                            params(move(rh.params)),
                            activations(move(rh.activations)),
                            inner_vals(move(rh.inner_vals)),
                            bias(move(rh.bias)),
                            fx_activations(move(rh.fx_activations)),
                            tmp_gradient(move(rh.tmp_gradient)),
                            containers(move(rh.containers)),
                            acum_pos(rh.acum_pos),
                            gradient_init(rh.gradient_init)

    {
    }

    void launch_forward(vector<T> &inputs)
    {
        myVec<T> ins(inputs.size(), CONTENT, &inputs);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                myVec<T> x = params[i] * ins;
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
            else
            {
                myVec<T> x = params[i] * inner_vals[i - 1];
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
        }
    }

    void init_gradient(vector<vector<T>> &set_ins, vector<vector<T>> &set_outs)
    {
        if (!gradient_init)
        {
            size_t ins_num = set_ins[0].size(); //* para guardar el tamaño de entradas, ya que el vector se hace 0 al moverlo
            vector<size_t> n_p_l;
            n_p_l.reserve(n_layers);
            acum_pos = set_ins.size();        //* acum_pos=n of sets
            containers.reserve(acum_pos + 1); //* para incluir al contenedor de acumulación
            fx_activations.reserve(n_layers);
            tmp_gradient.reserve(n_layers);

            for (int i = 0; i < n_layers; i++)
            {
                n_p_l.emplace_back(bias[i].size());
                fx_activations.emplace_back(n_p_l[i], CERO);
                tmp_gradient.emplace_back(n_p_l[i], CERO);
            }

            for (int i = 0; i < acum_pos; i++)
                containers.emplace_back(n_p_l, set_ins[i], set_outs[i]);

            containers.emplace_back(n_p_l, ins_num); //* contenedor de acumulación
            gradient_init = true;
        }
        else
            cout << "gradient already init!" << endl;
    }

    void launch_gradient(int iterations)
    {
        if (gradient_init)
        {
            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < acum_pos; j++)
                {
                    gradient(containers[j]);
                    containers[acum_pos] += containers[j];
                }

                containers[acum_pos].normalize_1();
                gradient_update_params(containers[acum_pos]);
                containers[acum_pos].reset();
            }
        }
        else
            cout << "initialize gradient!" << endl;
    }

    void print_params()
    {
        for (int i = 0; i < n_layers; i++)
        {
            cout << "Parámetros capa " << i << endl
                 << endl;
            params[i].print();
            cout << "Bias " << endl
                 << endl;
            bias[i].print();
        }
    }

    void print_inner_vals()
    {
        cout << "Valores internos" << endl
             << endl;
        for (auto &i : inner_vals)
            i.print();
    }

    void print_fx_activations()
    {

        cout << "fx activations" << endl
             << endl;
        for (auto &i : fx_activations)
            i.print();
    }

    myVec<T> &get_output()
    {
        return inner_vals.back();
    }

    size_t get_n_layers() const
    {
        return n_layers;
    }

    T get_params(size_t i, size_t j, size_t k)
    {
#ifdef ASSERT
        if (i < params.size())
            return params[i][j][k];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
#else
        return params[i][j][k];
#endif
    }

    T get_bias(size_t i, size_t j)
    {
#ifdef ASSERT
        if (i < bias.size())
            return bias[i][j];

        cout << "invalid access" << endl;
        exit(EXIT_FAILURE);
#else
        return bias[i][j];
#endif
    }
};
#endif