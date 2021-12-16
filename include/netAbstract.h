#ifndef NETABSTRACT_H
#define NETABSTRACT_H

#include <defines.h>

namespace net
{
    class net_abstract
    {
    public:
        virtual net_data get_net_data() = 0;
        virtual std::vector<DATA_TYPE> launch_forward(std::vector<DATA_TYPE> inputs) = 0; //* returns result, inputs como copia para mantener operaciones move
        virtual void init_gradient(net_sets sets) = 0;                                    //* net_sets como copia para mantener operaciones move
        virtual std::vector<DATA_TYPE> launch_gradient(int iterations) = 0;               //* returns it times error
        virtual void print_inner_vals() = 0;
        virtual signed long get_gradient_performance() = 0;
        virtual signed long get_forward_performance() = 0;
    };
}
#endif