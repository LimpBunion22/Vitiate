#include <netHandler.h>
#include <netCPU.h>
#include <netGPU.h>
#include <iostream>
// #include <netFPGA.h>

namespace net
{
    using namespace std;

    void net_handler::set_active_net(const string &net_key)
    {
        if (nets.find(net_key) == nets.end())
            cout << "net " << net_key << " doesn't exist!\n";
        else
        {
            active_net = nets[net_key].get();
            active_net_name = net_key;
        }
    }

    void net_handler::net_create_random_from_vector(const string &net_key, size_t implementation, size_t n_ins, const vector<size_t> &n_p_l)
    {
        switch (implementation)
        {
        case CUDA:
            if (nets.find(net_key) != nets.end())
            {
                // cout << "net " << net_key << " already exists, overwriting!\n";
                nets.erase(net_key);
                implementations.erase(net_key);
            }

            nets[net_key] = unique_ptr<net_abstract>(new gpu::net_gpu(n_ins, n_p_l));
            implementations[net_key] = implementation;
            break;
        case CPU:
        default:
            if (nets.find(net_key) != nets.end())
            {
                // cout << "net " << net_key << " already exists, overwriting!\n";
                nets.erase(net_key);
                implementations.erase(net_key);
            }

            nets[net_key] = unique_ptr<net_abstract>(new cpu::net_cpu(n_ins, n_p_l));
            implementations[net_key] = implementation;
            break;
        }
    }

    void net_handler::net_create(const string &net_key, size_t implementation, bool derivate, bool random, const string &file, bool file_reload)
    {
        switch (implementation)
        {
            bool succeeded;
        // case FPGA:
        //     if (random)
        //         succeeded = manager.load_net_structure(file, file_reload);
        //     else
        //         succeeded = manager.load_net(file, file_reload);

        //     if (succeeded)
        //     {
        //         if (nets.find(net_key) != nets.end())
        //         {
        //             // cout << "net " << net_key << " already exists, overwriting!\n";
        //             nets.erase(net_key);
        //             implementations.erase(net_key);
        //         }

        //         nets[net_key] = unique_ptr<net_abstract>(new fpga::net_fpga(manager.data, derivate, random));
        //         implementations[net_key] = implementation;
        //     }
        //     else
        //         cout << "failed to create new net " << net_key << " from file \"" << file << "\"\n";
        //     break;
        case CPU:
        default:
            if (random)
                succeeded = manager.load_net_structure(file, file_reload);
            else
                succeeded = manager.load_net(file, file_reload);

            if (succeeded)
            {
                if (nets.find(net_key) != nets.end())
                {
                    // cout << "net " << net_key << " already exists, overwriting!\n";
                    nets.erase(net_key);
                    implementations.erase(net_key);
                }

                nets[net_key] = unique_ptr<net_abstract>(new cpu::net_cpu(manager.data, derivate, random));
                implementations[net_key] = implementation;
            }
            else
                cout << "failed to create new net " << net_key << " from file \"" << file << "\"\n";
            break;
        }
    }

    vector<DATA_TYPE> net_handler::active_net_launch_forward(const vector<DATA_TYPE> &inputs)
    {
        if (!active_net)
        {
            cout << "no active net!\n";
            return vector<DATA_TYPE>{(DATA_TYPE)-1};
        }
        else
        {
            // cout << "launching net " << active_net_name << " forward! Outputs are:\n";
            return active_net->launch_forward(inputs);
        }
    }

    void net_handler::active_net_init_gradient(const string &file, bool file_reload)
    {
        if (!active_net)
            cout << "no active net!\n";
        else
        {
            bool succeeded = manager.load_sets(file, file_reload);

            if (succeeded)
            {
                active_net->init_gradient(manager.sets);
                // cout << "net " << active_net_name << " gradient initialized!\n";
            }
            else
                cout << "failed to initialize net " << active_net_name << " from file \"" << file << "\"\n";
        }
    }

    vector<DATA_TYPE> net_handler::active_net_launch_gradient(int iterations, DATA_TYPE error_threshold, DATA_TYPE multiplier)
    {
        if (!active_net)
        {
            cout << "no active net!\n";
            return vector<DATA_TYPE>{(DATA_TYPE)-1};
        }
        else
        {
            // cout << "launching net " << active_net_name << " gradient! Errors are:\n";
            return active_net->launch_gradient(iterations, error_threshold, multiplier);
        }
    }

    void net_handler::active_net_print_inner_vals()
    {
        if (!active_net)
            cout << "no active net!\n";
        else
        {
            cout << "printing net " << active_net_name << " inner vals\n";
            active_net->print_inner_vals();
        }
    }

    signed long net_handler::active_net_get_gradient_performance()
    {
        if (!active_net)
        {
            cout << "no active net!\n";
            return -1;
        }
        else
        {
            // cout << "net " << active_net_name << " gradient performance in us\n";
            return active_net->get_gradient_performance();
        }
    }

    signed long net_handler::active_net_get_forward_performance()
    {
        if (!active_net)
        {
            cout << "no active net!\n";
            return -1;
        }
        else
        {
            // cout << "net " << active_net_name << " forward performance in us\n";
            return active_net->get_forward_performance();
        }
    }

    void net_handler::active_net_write_net_to_file(const string &file)
    {
        if (!active_net)
            cout << "no active net!\n";
        else
            manager.write_net_to_file(file, active_net->get_net_data());
    }

    void net_handler::filter_image(const image_set &set)
    {
        if (!active_net)
            cout << "no active net!\n";
        else
            active_net->filter_image(set);
    }

    image_set net_handler::get_filtered_image()
    {
        if (!active_net)
        {
            cout << "no active net!\n";
            return image_set{.resized_image_data = {0}, .original_x_pos = 0, .original_y_pos = 0, .original_h = 0, .original_w = 0};
        }
        else
            return active_net->get_filtered_image();
    }
}
