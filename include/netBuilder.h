#ifndef NETBUILDER_H
#define NETBUILDER_H

#include <defines.h>

namespace net
{
    class builder
    {
    public:
        virtual builder *clone() const { return nullptr; }
        virtual ~builder() {}

        virtual void set_input_size(int input_size) = 0;
        virtual void build_fully_layer(int layer_size, int activation = net::RELU2) = 0;
        virtual void build_net() = 0;
        virtual void build_net_from_file(const net::layout &layout) = 0;
        virtual void build_net_from_data(int input_size, const std::vector<int> &n_p_l, const std::vector<int> &activations) = 0;
        virtual builder &attr(int attr, float value) = 0;
        virtual builder &attr(int attr, int value = 0) = 0;
        virtual std::vector<float> run_gradient(const net::set &set) = 0;
        virtual std::vector<float> run_forward(const std::vector<float> &input) = 0;
        virtual signed long get_gradient_performance() const = 0;
        virtual signed long get_forward_performance() const = 0;
        virtual net::layout get_net_data() const = 0;
        virtual void mutate(float limit) {}
    };
}
#endif