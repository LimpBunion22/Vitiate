#include <netHandler.h>
#include <netCPU.h>
#include <iostream>

namespace net
{
    using namespace std;

    void net_handler::net_create(const std::string &net_key, size_t implementation, bool derivate, bool random, const std::string &file, bool file_reload)
    {
        if (nets.find(net_key) == nets.end())
            switch (implementation)
            {
            case CPU:
            default:
                bool succeeded;

                if (random)
                    succeeded = manager.load_net_structure(file, file_reload);
                else
                    succeeded = manager.load_net(file, file_reload);

                if (succeeded)
                    nets[net_key] = unique_ptr<net_abstract>(new cpu::net_cpu(manager.data, derivate, random));
                else
                    cout << "failed to create new net from file \"" << file << "\"\n";
                break;
            }
        else
            cout << "net already exists!\n";
    }

    vector<DATA_TYPE> net_handler::launch_forward(const std::string &net_key, const vector<DATA_TYPE> &inputs)
    {
        if (nets.find(net_key) == nets.end())
        {
            cout << "net doesn't exist!\n";
            return vector{(DATA_TYPE)-1};
        }
        else
            return nets[net_key]->launch_forward(inputs);
    }

    void net_handler::init_gradient(const std::string &net_key, const std::string &file, bool file_reload)
    {
        if (nets.find(net_key) == nets.end())
            cout << "net doesn't exist!\n";
        else
        {
            bool succeeded = manager.load_sets(file, file_reload);
            
            if (succeeded)
                nets[net_key]->init_gradient(manager.sets);
            else
                cout << "failed to initialize net from file \"" << file << "\"\n";
        }
    }

    vector<DATA_TYPE> net_handler::launch_gradient(const std::string &net_key, int iterations)
    {
        if (nets.find(net_key) == nets.end())
        {
            cout << "net doesn't exist!\n";
            return vector{(DATA_TYPE)-1};
        }
        else
            return nets[net_key]->launch_gradient(iterations);
    }

    void net_handler::print_inner_vals(const std::string &net_key)
    {
        if (nets.find(net_key) == nets.end())
            cout << "net doesn't exist!\n";
        else
            nets[net_key]->print_inner_vals();
    }

    signed long net_handler::get_gradient_performance(const std::string &net_key)
    {
        if (nets.find(net_key) == nets.end())
        {
            cout << "net doesn't exist!\n";
            return -1;
        }
        else
            return nets[net_key]->get_gradient_performance();
    }

    signed long net_handler::get_forward_performance(const std::string &net_key)
    {
        if (nets.find(net_key) == nets.end())
        {
            cout << "net doesn't exist!\n";
            return -1;
        }
        else
            return nets[net_key]->get_forward_performance();
    }

    void net_handler::write_net_to_file(const std::string &net_key, const std::string &file)
    {
        if (nets.find(net_key) == nets.end())
            cout << "net doesn't exist!\n";
        else
            manager.write_net_to_file(file, nets[net_key]->get_net_data());
    }
}
