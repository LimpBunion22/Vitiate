#ifndef NETWORK_H
#define NETWORK_H

#include <mathStructs.h>
#include <chrono>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace chrono;

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
            : n_layers(n_p_l.size()), ins(ins), outs(outs)
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
                cout << "invalid dimensions lh is " << n_layers << " rh is " << rh.n_layers << "\n";
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
                cout << "invalid dimensions lh is " << n_layers << " rh is " << rh.n_layers << "\n";
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
                cout << "Gradiente parámetros capa " << i << "\n\n";
                fx_params[i].print();
                cout << "\n";
                cout << "Gradiente Bias\n\n";
                fx_bias[i].print();
                cout << "\n";
            }
        }
    };

private:
    size_t n_ins;
    size_t n_layers;
    vector<size_t> n_p_l;
    vector<myMatrix<T>> params;
    vector<myVec<myFun<T>>> activations;
    vector<myVec<T>> inner_vals;
    vector<myVec<T>> bias;
    vector<myVec<T>> fx_activations;
    vector<myVec<T>> tmp_gradient;
    vector<fx_container> containers;
    size_t acum_pos;
    bool gradient_init;
    int64_t gradient_performance;
    int64_t forward_performance;

public:
    class file_manager
    {
    private:
        const string HOME = getenv("HOME") ? getenv("HOME") : ".";

    public:
        size_t n_ins = 0;
        vector<size_t> n_p_l;
        vector<myMatrix<T>> params;
        vector<myVec<T>> bias;
        vector<vector<T>> set_ins;
        vector<vector<T>> set_outs;

    public:
        void load_net_structure(string name)
        {
            ifstream file_handler(HOME + "/" + name + ".csv", ios::in);

            if (file_handler.is_open())
            {
                string val, line;
                n_p_l.clear();
                getline(file_handler, line);
                stringstream s(line);
                getline(s, val, SEPARATOR);
                n_ins = (size_t)stoi(val);

                while (getline(s, val, SEPARATOR))
                    n_p_l.push_back((size_t)stoi(val));

                file_handler.close();
            }
            else
                cout << "unable to open file\n";
        }

        void load_net(string name)
        {
            load_net_structure(name);
            ifstream file_handler(HOME + "/" + name + ".csv", ios::in);

            if (file_handler.is_open())
            {
                string val, line;
                params.clear();
                bias.clear();
                params.reserve(n_p_l.size());
                bias.reserve(n_p_l.size());
                auto skip_lines = [&](int n_lines)
                {
                    for (int i = 0; i < n_lines; i++)
                        getline(file_handler, line);
                };

                skip_lines(2);

                for (int i = 0; i < n_p_l.size(); i++)
                {
                    if (i == 0)
                        params.emplace_back(n_p_l[i], n_ins, CERO);
                    else
                        params.emplace_back(n_p_l[i], n_p_l[i - 1], CERO);

                    bias.emplace_back(n_p_l[i], CERO);

                    for (int j = 0; j < params[i].rows(); j++)
                    {
                        getline(file_handler, line);
                        stringstream s(line);
                        size_t k = 0;

                        while (getline(s, val, SEPARATOR))
                            params[i][j][k++] = (T)stod(val);
                    }

                    skip_lines(1);
                    getline(file_handler, line);
                    stringstream s(line);
                    size_t j = 0;

                    while (getline(s, val, SEPARATOR))
                        bias[i][j++] = (T)stod(val);

                    skip_lines(1);
                }

                file_handler.close();
            }
            else
                cout << "unable to open file\n";
        }

        void load_sets(string name)
        {
            ifstream file_handler(HOME + "/" + name + ".csv", ios::in);

            if (file_handler.is_open())
            {
                string val, line;
                set_ins.clear();
                set_outs.clear();
                auto skip_lines = [&](int n_lines)
                {
                    for (int i = 0; i < n_lines; i++)
                        getline(file_handler, line);
                };

                getline(file_handler, line);
                stringstream s(line);
                getline(s, val, SEPARATOR);
                size_t n_sets = (size_t)stoi(val);
                skip_lines(1);
                set_ins.reserve(n_sets);
                set_outs.reserve(n_sets);

                for (int i = 0; i < n_sets; i++)
                {
                    set_ins.emplace_back(n_ins, 0);
                    set_outs.emplace_back(n_p_l[n_p_l.size() - 1], 0);

                    getline(file_handler, line);

                    {
                        stringstream s(line);
                        size_t j = 0;

                        while (getline(s, val, SEPARATOR))
                            set_ins[i][j++] = (T)stod(val);
                    }

                    getline(file_handler, line);

                    {
                        stringstream s(line);
                        size_t j = 0;

                        while (getline(s, val, SEPARATOR))
                            set_outs[i][j++] = (T)stod(val);
                    }

                    skip_lines(1);
                }

                file_handler.close();
            }
            else
                cout << "unable to open file\n";
        }

        //^ last row element also followed by separator char!!
        void write_net_to_file(string name, network &net)
        {
            ofstream file_handler(HOME + "/" + name + ".csv", ios::out | ios::trunc);

            if (file_handler.is_open())
            {
                file_handler << net.n_ins << " ";

                for (auto &i : net.n_p_l)
                    file_handler << i << " ";

                file_handler << "\n\n";

                for (int i = 0; i < net.n_layers; i++)
                {
                    for (int j = 0; j < net.params[i].rows(); j++)
                    {
                        for (int k = 0; k < net.params[i].cols(); k++)
                            file_handler << net.params[i][j][k] << " ";

                        file_handler << "\n";
                    }

                    file_handler << "\n";

                    for (int j = 0; j < net.bias[i].size(); j++)
                        file_handler << net.bias[i][j] << " ";

                    file_handler << "\n\n";
                }

                file_handler.close();
            }
            else
                cout << "unable to open file\n";
        }
    };

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
    myVec<T> gradient(fx_container &fx) //* returns R vec
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

        return R;
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
    network(file_manager &manager, bool derivate, bool random)
        : n_layers(manager.n_p_l.size()), acum_pos(0), gradient_init(false), gradient_performance(0), forward_performance(0), n_p_l(manager.n_p_l), n_ins(manager.n_ins)

    {
        params.reserve(n_layers);
        activations.reserve(n_layers);
        inner_vals.reserve(n_layers);
        bias.reserve(n_layers);

        if (random)
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
        else
            for (int i = 0; i < n_layers; i++)
            {
                if (i == 0)
                    params.emplace_back(manager.params[i]);
                else
                    params.emplace_back(manager.params[i]);

                activations.emplace_back(n_p_l[i], derivate);
                inner_vals.emplace_back(n_p_l[i], CERO);
                bias.emplace_back(manager.bias[i]);
            }
    }

    network(size_t n_ins, vector<size_t> &n_p_l, bool derivate)
        : n_layers(n_p_l.size()), acum_pos(0), gradient_init(false), gradient_performance(0), forward_performance(0), n_p_l(n_p_l), n_ins(n_ins)

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

    network(size_t n_ins, vector<size_t> &n_p_l, vector<vector<vector<T>>> &p, vector<vector<T>> &b, bool derivate)
        : n_layers(n_p_l.size()), acum_pos(0), gradient_init(false), gradient_performance(0), forward_performance(0), n_p_l(n_p_l), n_ins(n_ins)
    {
        params.reserve(n_layers);
        activations.reserve(n_layers);
        inner_vals.reserve(n_layers);
        bias.reserve(n_layers);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
                params.emplace_back(p[i]);
            else
                params.emplace_back(p[i]);

            activations.emplace_back(n_p_l[i], derivate);
            inner_vals.emplace_back(n_p_l[i], CERO);
            bias.emplace_back(b[i]);
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
            gradient_performance = rh.gradient_performance;
            forward_performance = rh.forward_performance;
            n_p_l = rh.n_p_l;
            n_ins = rh.n_ins;
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
            gradient_performance = rh.gradient_performance;
            forward_performance = rh.forward_performance;
            n_p_l = move(rh.n_p_l);
            n_ins = rh.n_ins;
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
                                 gradient_init(rh.gradient_init),
                                 gradient_performance(rh.gradient_performance),
                                 forward_performance(rh.forward_performance),
                                 n_p_l(rh.n_p_l),
                                 n_ins(rh.n_ins)

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
                            gradient_init(rh.gradient_init),
                            gradient_performance(rh.gradient_performance),
                            forward_performance(rh.forward_performance),
                            n_p_l(move(rh.n_p_l)),
                            n_ins(rh.n_ins)

    {
    }

    vector<T> launch_forward(vector<T> &inputs) //* returns result
    {
#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif
        myVec<T> ins(inputs);

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
#ifdef PERFORMANCE
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        forward_performance = duration.count();
#endif

        return inner_vals.back().copy_inner_vec();
    }

    void init_gradient(vector<vector<T>> &set_ins, vector<vector<T>> &set_outs)
    {
        if (!gradient_init)
        {
            size_t ins_num = set_ins[0].size(); //* para guardar el tamaño de entradas, ya que el vector se hace 0 al moverlo
            acum_pos = set_ins.size();          //* acum_pos=n of sets
            containers.reserve(acum_pos + 1);   //* para incluir al contenedor de acumulación
            fx_activations.reserve(n_layers);
            tmp_gradient.reserve(n_layers);

            for (int i = 0; i < n_layers; i++)
            {
                fx_activations.emplace_back(n_p_l[i], CERO);
                tmp_gradient.emplace_back(n_p_l[i], CERO);
            }

            for (int i = 0; i < acum_pos; i++)
                containers.emplace_back(n_p_l, set_ins[i], set_outs[i]);

            containers.emplace_back(n_p_l, ins_num); //* contenedor de acumulación
            gradient_init = true;
        }
        else
            cout << "gradient already init!\n";
    }

    void init_gradient(file_manager &manager)
    {
        init_gradient(manager.set_ins, manager.set_outs);
    }

    vector<T> launch_gradient(int iterations) //* returns it times errors
    {
        if (gradient_init)
        {
#ifdef PERFORMANCE
            auto start = high_resolution_clock::now();
#endif
            vector<T> set_errors(iterations, 0);
            myVec<T> set_single_errors(acum_pos, CERO);

            for (int i = 0; i < iterations; i++)
            {
                for (int j = 0; j < acum_pos; j++)
                {
                    set_single_errors[j] = gradient(containers[j]).elems_abs().reduce();
                    containers[acum_pos] += containers[j];
                }

                containers[acum_pos].normalize_1();
                gradient_update_params(containers[acum_pos]);
                containers[acum_pos].reset();
                set_errors[i] = set_single_errors.reduce();
            }
#ifdef PERFORMANCE
            auto end = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(end - start);
            gradient_performance = duration.count();
#endif
            return set_errors;
        }
        else
        {
            cout << "initialize gradient!\n";
            return vector<T>(iterations, 0);
        }
    }

    void print_params()
    {
        for (int i = 0; i < n_layers; i++)
        {
            cout << "Parámetros capa " << i << "\n\n";
            params[i].print();
            cout << "\n";
            cout << "Bias\n\n";
            bias[i].print();
            cout << "\n";
        }
    }

    void print_inner_vals()
    {
        cout << "Valores internos\n\n";

        for (auto &i : inner_vals)
        {
            i.print();
            cout << "\n";
        }
    }

    int64_t get_gradient_performance()
    {
#ifdef PERFORMANCE
        return gradient_performance;
#else
        cout << "performance not enabled\n";
        return 0;
#endif
    }

    int64_t get_forward_performance()
    {
#ifdef PERFORMANCE
        return forward_performance;
#else
        cout << "performance not enabled\n";
        return 0;
#endif
    }
};
#endif