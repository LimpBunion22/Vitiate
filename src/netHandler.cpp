#include <iostream>
#include <netHandler.h>
#include <builderCPU.h>
#ifdef USE_FPGA
#include <netFPGA.h>
#include <experimental/filesystem>
#endif
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio_c.h>

namespace net
{
    // management
    void handler::set_active_net(const std::string &key)
    {
        if (_nets.find(key) == _nets.end())
        {
            std::cout << YELLOW << "net " << key << " doesn't exist" << RESET << "\n";
            return;
        }

        _active_net = _nets[key].get();
        _active_net_name = key;
    }

    void handler::delete_net(const std::string &key)
    {
        if (_nets.find(key) == _nets.end())
        {
            std::cout << YELLOW << "can't delete nonexistent net " << key << RESET << "\n";
            return;
        }

        if (_active_net_name == key)
        {
            _active_net = nullptr;
            _active_net_name = " ";
        }

        _nets.erase(key);
    }

    void handler::instantiate(const std::string &key, int implementation)
    {
        if (_nets.find(key) != _nets.end())
        {
            _nets.erase(key);
            _implementations.erase(key);
        }

        switch (implementation)
        {
        case GPU:
            _nets[key] = std::make_shared<gpu::gpu_builder>(_cub, _cublas, _stream);
            _implementations[key] = implementation;
            break;
        case CPU:
            _nets[key] = std::make_shared<cpu::cpu_builder>();
            _implementations[key] = implementation;
            break;
#ifdef USE_FPGA
        case FPGA:
            _nets[key] = unique_ptr<net_abstract>(new fpga::net_fpga());
            implementations[key] = implementation;
            if (mustang_handler_init == false)
            {
                // cout << BLUE << "handler: Activating  handler" << RESET << "\n";
                mustang_handler_init = true;
                mustang_handler.activate_handler();
                // cout << BLUE << "handler: Handler activated" << RESET << "\n";
            }

            // cout << BLUE << "handler: Creating fpga net" << RESET << "\n";
            nets[net_key] = unique_ptr<net_abstract>(new fpga::net_fpga(n_ins, n_p_l, activation_type, mustang_handler));
            // cout << BLUE << "handler: FPGA net created" << RESET << "\n";
            implementations[net_key] = implementation;
            break;
#endif
        default:
            std::cout << RED << "implementation doesn't exist" << RESET << "\n";
            break;
        }
    }

    void handler::set_input_size(int input_size)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }
    }

    void handler::build_fully_layer(int layer_size, int activation)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }

        _active_net->build_fully_layer(layer_size, activation);
    }

    void handler::build_net()
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }

        _active_net->build_net();
    }

    void handler::build_net_from_file(const std::string &file, bool file_reload)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }

        if (_file_manager.load_net(file, file_reload))
            return _active_net->build_net_from_file(_file_manager._net);

        std::cout << RED << "failed to build net " << _active_net_name << " from file \"" << file << '\"' << RESET "\n";
    }

    handler &handler::attr(int attr, float value)
    {
        if (!_active_net)
            std::cout << YELLOW << "no active net" << RESET << "\n ";
        else
            _active_net->attr(attr, value);

        return *this;
    }

    handler &handler::attr(int attr, int value)
    {
        if (!_active_net)
            std::cout << YELLOW << "no active net" << RESET << "\n ";
        else
            _active_net->attr(attr, value);

        return *this;
    }

    // run methods
    std::vector<float> handler::run_forward(const std::vector<float> &input)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return {-1.0f};
        }

        return _active_net->run_forward(input);
    }

    std::vector<float> handler::run_gradient(const net::set &set)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return {-1.0f};
        }

        return _active_net->run_gradient(set);
    }

    std::vector<float> handler::run_gradient(const std::string &file, bool file_reload)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return {-1.0f};
        }

        if (_file_manager.load_set(file, file_reload))
            return _active_net->run_gradient(_file_manager._set);

        std::cout << RED << "failed to run gradient from file \"" << file << '\"' << RESET "\n";
        return {-1.0f};
    }

    // metrics
    signed long handler::get_gradient_performance() const
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return -1;
        }

        return _active_net->get_gradient_performance();
    }

    signed long handler::get_forward_performance() const
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return -1;
        }

        return _active_net->get_forward_performance();
    }

    // disk
    void handler::write_net_to_file(const std::string &file)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }

        _file_manager.write_net_to_file(file, _active_net->get_net_data());
    }

    void handler::write_set_to_file(const std::string &file, const set &set)
    {
        _file_manager.write_set_to_file(file, set);
    }

// fpga
#ifdef USE_FPGA
    void handler::enq_fpga_net(const std::string &net_key, const std::vector<float> &inputs, bool reload, bool same_in, bool big_nets)
    {
        if (nets.find(net_key) == nets.end())
            cout << YELLOW << "net " << net_key << " doesn't exist" << RESET << "\n";
        else
        {
            if (implementations[net_key] != FPGA)
                cout << YELLOW << "net " << net_key << " is not a fpga net" << RESET << "\n";
            else
            {
                fpga::net_fpga *enq_net = dynamic_cast<fpga::net_fpga *>(nets[net_key].get());
                enq_net->enqueue_net(inputs, reload, same_in, big_nets);
            }
        }
    }

    void handler::exe_fpga_nets()
    {

        if (implementations[active_net_name] != FPGA)
            cout << YELLOW << "active net " << active_net_name << " is not a fpga net" << RESET << "\n";
        else
        {
            fpga::net_fpga *enq_net = dynamic_cast<fpga::net_fpga *>(nets[active_net_name].get());
            enq_net->solve_pack();
        }
    }

    std::vector<float> handler::read_fpga_net(const std::string &net_key)
    {
        if (nets.find(net_key) == nets.end())
            cout << YELLOW << "net " << net_key << " doesn't exist" << RESET << "\n";
        else
        {
            if (implementations[net_key] != FPGA)
                cout << YELLOW << "net " << net_key << " is not a fpga net" << RESET << "\n";
            else
            {
                fpga::net_fpga *enq_net = dynamic_cast<fpga::net_fpga *>(nets[net_key].get());
                return enq_net->read_net();
            }
        }
    }
#endif

    // ctors/dtors
    handler::handler(const std::string &path) : _file_manager(path), _active_net(nullptr), _stream(gpu::create_stream())
    {
    }

    handler::~handler()
    {
        gpu::destroy_stream(_stream);
        gpu::cub_free(_cub);
        gpu::cublas_free(_cublas);
    }
}
