#ifndef NETWORK_H
#define NETWORK_H

#include <defines.h>
#include <map>
#include <iostream>
#include <memory>

class file_manager;
class net_abstract;

class net_handler
{
private:
    std::map<std::string, std::unique_ptr<net_abstract>> nets;

public:
    void net_create(size_t net_key, size_t implementation, file_manager &manager, bool derivate, bool random);
    std::vector<DATA_TYPE> launch_forward(size_t net_key, std::vector<DATA_TYPE> &inputs); //* returns result
    void init_gradient(size_t net_key, file_manager &manager);
    std::vector<DATA_TYPE> launch_gradient(size_t net_key, int iterations); //* returns it times error
    void print_inner_vals(size_t net_key);
    signed long get_gradient_performance(size_t net_key);
    signed long get_forward_performance(size_t net_key);
};
#endif