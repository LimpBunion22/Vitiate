#ifndef BUILDERCPU_H
#define BUILDERCPU_H

#include <memoryCPU.h>
#include <normCPU.h>
#include <memory>
#include <algosCPU.h>
#include <layersCPU.h>
#include <chrono>
#include <netBuilder.h>

namespace cpu
{
    class cpu_builder : public net::builder
    {
    private:
        typedef struct
        {
            int size;
            int activation;
        } _layer_layout;

    private:
        // schedulers
        fully_con_scheduler _f_scheduler;
        gradient_scheduler _g_scheduler;
        batch_scheduler _b_scheduler;
        int _batch_size;

        // layers
        std::vector<std::unique_ptr<fully_layer>> _f_layers;
        out_fully_layer *_out_layer;
        in_fully_layer *_in_layer;
        std::vector<_layer_layout> _net_layout;
        int _input_size;

        // norm/gradient update
        float _alpha;
        float _alpha_decay;
        float _error_threshold;
        single::gradient_norm _g_norm;
        int _epochs;

        // algos
        std::unique_ptr<single::adam> _adam;
        std::unique_ptr<single::dropout> _dropout;

        // performance
        int64_t _gradient_performance = 0;
        int64_t _forward_performance = 0;

        // control
        bool _net_built;
        bool _gradient_built;

    private:
        bool _build_net();
        void _build_gradient();
        void _gradient();

    public:
        cpu_builder();
        cpu_builder(const cpu_builder &rh);
        cpu_builder(cpu_builder &&rh);
        cpu_builder &operator=(const cpu_builder &rh) = delete;
        cpu_builder &operator=(cpu_builder &&rh) = delete;
        net::builder *clone() const override;
        ~cpu_builder() = default;

        void set_input_size(int input_size) override;
        void build_fully_layer(int layer_size, int activation = net::RELU2) override;
        void build_net() override;
        void build_net_from_file(const net::layout &layout) override;
        void build_net_from_data(int input_size, const std::vector<int> &n_p_l, const std::vector<int> &activations) override;
        net::builder &attr(int attr, float value) override;
        net::builder &attr(int attr, int value = 0) override;
        std::vector<float> run_gradient(const net::set &set) override;
        std::vector<float> run_forward(const std::vector<float> &input) override;
        void run_forward(float *input, float *output);
        signed long get_gradient_performance() const override;
        signed long get_forward_performance() const override;
        net::layout get_net_data() const override;

        // void mutate(float percentage);
        void mutate(float limit) override;
    };
}

#endif