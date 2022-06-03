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
            _nets[key] = std::make_shared<gpu::gpu_builder>(_cub, _cublas, _streams);
            _implementations[key] = implementation;
            break;
        case CPU:
            _nets[key] = std::make_shared<cpu::cpu_builder>();
            _implementations[key] = implementation;
            break;
#ifdef USE_FPGA
        case FPGA:
            if (_mustang_handler_init == false)
            {
                // std::cout << BLUE << "handler: Activating  handler" << RESET << "\n";
                _mustang_handler_init = true;
                _mustang_handler.activate_handler();
                // std::cout << BLUE << "handler: Handler activated" << RESET << "\n";
            }

            // std::cout << BLUE << "handler: Creating fpga net" << RESET << "\n";
            _nets[key] = std::make_unique<fpga::net_fpga>(_mustang_handler);
            // std::cout << BLUE << "handler: FPGA net created" << RESET << "\n";
            _implementations[key] = implementation;
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

        _active_net->set_input_size(input_size);
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

    void handler::build_net_from_data(int input_size, const std::vector<int> &n_p_l, const std::vector<int> &activations)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }

        _active_net->build_net_from_data(input_size, n_p_l, activations);
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
    std::vector<float> handler::run_forward(const std::vector<float> &input) // Mirar como pasar big nets y reload a run forward
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

    void handler::mutate(float limit)
    {
        if (!_active_net)
        {
            std::cout << YELLOW << "no active net" << RESET << "\n ";
            return;
        }

        _active_net->mutate(limit);
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
    void handler::enq_fpga_net(const std::string &key, const std::vector<float> &inputs, bool reload, bool same_in, bool big_nets)
    {
        if (_nets.find(key) == _nets.end())
        {
            std::cout << YELLOW << "net " << key << " doesn't exist" << RESET << "\n";
            return;
        }

        if (_implementations[key] != FPGA)
            std::cout << YELLOW << "net " << key << " is not a fpga net" << RESET << "\n";
        else
        {
            fpga::net_fpga *enq_net = dynamic_cast<fpga::net_fpga *>(_nets[key].get());
            enq_net->enqueue_net(inputs, reload, same_in, big_nets);
        }
    }

    void handler::exe_fpga_nets()
    {
        if (_implementations[_active_net_name] != FPGA)
            std::cout << YELLOW << "active net " << _active_net_name << " is not a fpga net" << RESET << "\n";
        else
        {
            fpga::net_fpga *enq_net = dynamic_cast<fpga::net_fpga *>(_nets[_active_net_name].get());
            enq_net->solve_pack();
        }
    }

    std::vector<float> handler::read_fpga_net(const std::string &key)
    {
        if (_nets.find(key) == _nets.end())
        {
            std::cout << YELLOW << "net " << key << " doesn't exist" << RESET << "\n";
            return {-1.0f};
        }

        if (_implementations[key] != FPGA)
        {
            std::cout << YELLOW << "net " << key << " is not a fpga net" << RESET << "\n";
            return {-1.0f};
        }

        fpga::net_fpga *enq_net = dynamic_cast<fpga::net_fpga *>(_nets[key].get());
        return enq_net->read_net();
    }
#endif

    // ctors/dtors
    handler::handler(const std::string &path) : _file_manager(path), _active_net(nullptr), _streams(1)
    {
        for (auto &i : _streams)
            i = gpu::create_stream();
    }

    handler::~handler()
    {
        for (auto &i : _streams)
            gpu::destroy_stream(i);

        gpu::cub_free(_cub);
        gpu::cublas_free(_cublas);
    }

    void handler::clone(const std::string &original, const std::string &clone)
    {
        if (_nets.find(original) == _nets.end())
        {
            std::cout << YELLOW << "net " << original << " to be cloned doesn't exist" << RESET << "\n";
            return;
        }

        if (original == clone)
        {
            std::cout << YELLOW << "original and clone cannot be equally named" << RESET << "\n";
            return;
        }

        _nets[clone].reset(_nets[original]->clone()); // reset automatically deletes any previously existing net so no need to manually erase
        _implementations[clone] = _implementations[original];
    }
}
