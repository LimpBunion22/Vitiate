#include <netCPU.h>
#include <math.h>
#include <iostream>

//* net_cpu::FX_CONTAINER
namespace cpu
{
    using namespace std;

    net_cpu::fx_container::fx_container(const vector<size_t> &n_p_l, size_t ins_num)
        : n_layers(n_p_l.size()), ins(1, CERO), outs(1, CERO)
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

    net_cpu::fx_container::fx_container(const vector<size_t> &n_p_l, const vector<DATA_TYPE> &ins, const vector<DATA_TYPE> &outs)
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

    net_cpu::fx_container::fx_container(fx_container &&rh) : n_layers(rh.n_layers),
                                                                 fx_params(move(rh.fx_params)),
                                                                 fx_bias(move(rh.fx_bias)),
                                                                 ins(move(rh.ins)),
                                                                 outs(move(rh.outs))
    {
    }

    void net_cpu::fx_container::reset()
    {
        for (int i = 0; i < n_layers; i++)
        {
            for (int j = 0; j < fx_params[i].rows(); j++)
            {
                for (int k = 0; k < fx_params[i].cols(); k++)
                    fx_params[i](j, k) = 0;

                fx_bias[i][j] = 0;
            }
        }
    }

    net_cpu::fx_container &net_cpu::fx_container::operator+=(net_cpu::fx_container &rh)
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

    net_cpu::fx_container &net_cpu::fx_container::operator-=(net_cpu::fx_container &rh)
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

    void net_cpu::fx_container::normalize_0(DATA_TYPE factor)
    {
        DATA_TYPE max = 0;
        DATA_TYPE val = 0;

        for (int i = 0; i < n_layers; i++)
        {
            for (int j = 0; j < fx_params[i].rows(); j++)
            {
                for (int k = 0; k < fx_params[i].cols(); k++)
                {
                    val = abs(fx_params[i](j, k));
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
                    fx_params[i](j, k) /= max;

                fx_bias[i][j] /= max;
            }
        }
    }

    void net_cpu::fx_container::normalize_1(DATA_TYPE factor)
    {
        DATA_TYPE max = 0;

        for (int i = 0; i < n_layers; i++)
        {
            for (int j = 0; j < fx_params[i].rows(); j++)
            {
                for (int k = 0; k < fx_params[i].cols(); k++)
                    max += abs(fx_params[i](j, k));

                max += abs(fx_bias[i][j]);
            }
        }

        max *= factor;

        for (int i = 0; i < n_layers; i++)
        {
            for (int j = 0; j < fx_params[i].rows(); j++)
            {
                for (int k = 0; k < fx_params[i].cols(); k++)
                    fx_params[i](j, k) /= max;

                fx_bias[i][j] /= max;
            }
        }
    }

    void net_cpu::fx_container::normalize_2(DATA_TYPE factor)
    {
        DATA_TYPE max = 0;

        for (int i = 0; i < n_layers; i++)
        {
            for (int j = 0; j < fx_params[i].rows(); j++)
            {
                for (int k = 0; k < fx_params[i].cols(); k++)
                    max += fx_params[i](j, k) * fx_params[i](j, k);

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
                    fx_params[i](j, k) /= max;

                fx_bias[i][j] /= max;
            }
        }
    }

    void net_cpu::fx_container::print_fx()
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
}

//* net_cpu
namespace cpu
{
    using namespace std;
    using namespace chrono;

    net_cpu::net_cpu(const net::net_data &data, bool derivate, bool random)
        : n_layers(data.n_p_l.size()), acum_pos(0), gradient_init(false), gradient_performance(0), forward_performance(0), n_p_l(data.n_p_l), n_ins(data.n_ins)

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
        {
            for (int i = 0; i < n_layers; i++)
            {
                if (i == 0)
                    params.emplace_back(data.params[i]);
                else
                    params.emplace_back(data.params[i]);

                activations.emplace_back(n_p_l[i], derivate);
                inner_vals.emplace_back(n_p_l[i], CERO);
                bias.emplace_back(data.bias[i]);
            }
        }
    }

    net_cpu::net_cpu(net_cpu &&rh) : n_layers(rh.n_layers),
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

    net_cpu &net_cpu::operator=(net_cpu &&rh)
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

    void net_cpu::forward_gradient(my_vec &ins)
    {
        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                my_vec x = params[i] * ins;
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
                fx_activations[i] = activations[i].derivate(x);
            }
            else
            {
                my_vec x = params[i] * inner_vals[i - 1];
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
                fx_activations[i] = activations[i].derivate(x);
            }
        }
    }

    //* R*(fx^P)*(fx^P)...^(fx matrix I)
    my_vec net_cpu::gradient(net_cpu::fx_container &fx) //* returns R vec
    //* IMPLEMENDATA_TYPEACIÓN
    {
        forward_gradient(fx.ins);
        my_vec R = fx.outs - inner_vals[n_layers - 1];

        //* caso salidas
        tmp_gradient[n_layers - 1] = R;
        fx.fx_params[n_layers - 1] = make_from(fx_activations[n_layers - 1], inner_vals[n_layers - 2]);
        fx.fx_params[n_layers - 1] ^= tmp_gradient[n_layers - 1];
        fx.fx_bias[n_layers - 1] = tmp_gradient[n_layers - 1] ^ fx_activations[n_layers - 1];

        //* caso general
        for (int i = n_layers - 2; i > 0; i--)
        {
            my_matrix fx_special_product_params = params[i + 1] ^ fx_activations[i + 1];
            tmp_gradient[i] = tmp_gradient[i + 1] * fx_special_product_params;
            fx.fx_params[i] = make_from(fx_activations[i], inner_vals[i - 1]);
            fx.fx_params[i] ^= tmp_gradient[i];
            fx.fx_bias[i] = tmp_gradient[i] ^ fx_activations[i];
        }

        //* caso entradas
        my_matrix fx_special_product_params = params[1] ^ fx_activations[1];
        tmp_gradient[0] = tmp_gradient[1] * fx_special_product_params;
        fx.fx_params[0] = make_from(fx_activations[0], fx.ins);
        fx.fx_params[0] ^= tmp_gradient[0];
        fx.fx_bias[0] = tmp_gradient[0] ^ fx_activations[0];

        return R;
    }

    void net_cpu::gradient_update_params(net_cpu::fx_container &fx)
    {
        for (int i = 0; i < n_layers; i++)
        {
            params[i] += fx.fx_params[i];
            bias[i] += fx.fx_bias[i];
        }
    }

    net::net_data net_cpu::get_net_data() // TODO:implementar
    {
        net::net_data data;
        data.n_ins = n_ins;
        data.n_layers = n_layers;
        data.n_p_l = n_p_l;

        for (int i = 0; i < n_layers; i++)
        {
            data.params.emplace_back(params[i].rows(), vector<DATA_TYPE>(params[i].cols(), 0));

            for (int j = 0; j < params[i].rows(); j++)
                for (int k = 0; k < params[i].cols(); k++)
                    data.params[i][j][k] = params[i](j, k);

            data.bias.emplace_back(bias[i].size(), 0);

            for (int j = 0; j < bias[i].size(); j++)
                data.bias[i][j] = bias[i][j];
        }

        return data;
    }

    vector<DATA_TYPE> net_cpu::launch_forward(const vector<DATA_TYPE> &inputs) //* returns result
    {
#ifdef PERFORMANCE
        auto start = high_resolution_clock::now();
#endif
        vector<DATA_TYPE> inputs_copy = inputs;
        my_vec ins(inputs_copy);

        for (int i = 0; i < n_layers; i++)
        {
            if (i == 0)
            {
                my_vec x = params[i] * ins;
                x += bias[i];
                inner_vals[i] = activations[i].calculate(x);
            }
            else
            {
                my_vec x = params[i] * inner_vals[i - 1];
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

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    void net_cpu::init_gradient(const net::net_sets &sets)
    {
        if (!gradient_init)
        {
            size_t ins_num = sets.set_ins[0].size(); //* para guardar el tamaño de entradas, ya que el vector se hace 0 al moverlo
            acum_pos = sets.set_ins.size();          //* acum_pos=n of sets
            containers.reserve(acum_pos + 1);        //* para incluir al contenedor de acumulación
            fx_activations.reserve(n_layers);
            tmp_gradient.reserve(n_layers);

            for (int i = 0; i < n_layers; i++)
            {
                fx_activations.emplace_back(n_p_l[i], CERO);
                tmp_gradient.emplace_back(n_p_l[i], CERO);
            }

            for (int i = 0; i < acum_pos; i++)
                containers.emplace_back(n_p_l, sets.set_ins[i], sets.set_outs[i]);

            containers.emplace_back(n_p_l, ins_num); //* contenedor de acumulación
            gradient_init = true;
        }
        else
            cout << "gradient already init!\n";
    }

    //^ HANDLER + IMPLEMENDATA_TYPEACIÓN (REVISAR MOVE OP)
    vector<DATA_TYPE> net_cpu::launch_gradient(int iterations) //* returns it times errors
    {
        if (gradient_init)
        {
#ifdef PERFORMANCE
            auto start = high_resolution_clock::now();
#endif
            vector<DATA_TYPE> set_errors(iterations, 0);
            my_vec set_single_errors(acum_pos, CERO);

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
            return vector<DATA_TYPE>(iterations, 0);
        }
    }

    void net_cpu::print_inner_vals()
    {
        cout << "Valores internos\n\n";

        for (auto &i : inner_vals)
        {
            i.print();
            cout << "\n";
        }
    }

    int64_t net_cpu::get_gradient_performance()
    {
#ifdef PERFORMANCE
        return gradient_performance;
#else
        cout << "performance not enabled\n";
        return 0;
#endif
    }

    int64_t net_cpu::get_forward_performance()
    {
#ifdef PERFORMANCE
        return forward_performance;
#else
        cout << "performance not enabled\n";
        return 0;
#endif
    }
}