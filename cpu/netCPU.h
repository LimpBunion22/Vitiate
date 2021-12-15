#ifndef NETCPU_H
#define NETCPU_H

#include <netAbstract.h>
#include <mathStructs.h>
#include <chrono>

namespace cpu
{
    class net_cpu : public net_abstract
    {
    private:
        class fx_container
        {
        private:
            size_t n_layers;

        public:
            std::vector<my_matrix> fx_params;
            std::vector<my_vec> fx_bias;
            my_vec ins;
            my_vec outs;

        public:
            fx_container(std::vector<size_t> &n_p_l, size_t ins_num);
            fx_container(std::vector<size_t> &n_p_l, std::vector<DATA_TYPE> &ins, std::vector<DATA_TYPE> &outs);
            fx_container(const fx_container &rh);
            fx_container(fx_container &&rh);
            fx_container &operator=(const fx_container &rh);

            fx_container &operator+=(fx_container &rh);
            fx_container &operator-=(fx_container &rh);

            void reset();
            void normalize_0(DATA_TYPE factor = 1);
            void normalize_1(DATA_TYPE factor = 1);
            void normalize_2(DATA_TYPE factor = 1);
            void print_fx();
        };

    private:
        size_t n_ins;
        size_t n_layers;
        std::vector<size_t> n_p_l;
        std::vector<my_matrix> params;
        std::vector<my_vec_fun> activations;
        std::vector<my_vec> inner_vals;
        std::vector<my_vec> bias;
        std::vector<my_vec> fx_activations;
        std::vector<my_vec> tmp_gradient;
        std::vector<fx_container> containers;
        size_t acum_pos;
        bool gradient_init;
        int64_t gradient_performance;
        int64_t forward_performance;

    private:
        net_cpu() = delete;

        void forward_gradient(my_vec &ins);
        my_vec gradient(net_cpu::fx_container &fx);
        void gradient_update_params(fx_container &fx);

    public:
        net_cpu(net_data data, bool derivate, bool random); //* net_data como copia para mantener operaciones move
        net_cpu(const net_cpu &rh);
        net_cpu(net_cpu &&rh);
        net_cpu &operator=(const net_cpu &rh);
        net_cpu &operator=(net_cpu &&rh);

        net_data get_net_data() override; // TODO::implementar
        std::vector<DATA_TYPE> launch_forward(std::vector<DATA_TYPE> &inputs) override;
        void init_gradient(net_sets sets) override;
        std::vector<DATA_TYPE> launch_gradient(int iterations) override;
        void print_inner_vals() override;
        signed long get_gradient_performance() override;
        signed long get_forward_performance() override;
    };
}

#endif