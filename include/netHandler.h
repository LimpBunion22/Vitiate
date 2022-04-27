#ifndef NETHANDLER_H
#define NETHANDLER_H

#include <defines.h>
#include <map>
#include <string>
#include <memory>
#include <netFileManager.h>
#include <netBuilder.h>
#include <builderGPU.h>

namespace net
{
    class builder;

    constexpr int CPU = 0;
    constexpr int GPU = 1;
#ifdef USE_FPGA
    constexpr int FPGA = 2;
#endif

    class handler
    {
    private:
        std::map<std::string, std::unique_ptr<builder>> _nets;
        std::map<std::string, int> _implementations;
        file_manager _file_manager;
        builder *_active_net;
        std::string _active_net_name;
        cudaStream_t _stream;
        CREATE_CUB_DATA(_cub);
        CREATE_CUBLAS_DATA(_cublas);

    public:
        handler(const std::string &path);
        ~handler();

        // management
        void set_active_net(const std::string &key);
        void delete_net(const std::string &key);
        void instantiate(const std::string &key, int implementation);
        void set_input_size(int input_size);
        void build_fully_layer(int layer_size, int activation = net::RELU2);
        void build_net();
        void build_net_from_file(const std::string &file, bool file_reload);
        handler &attr(int attr, float value);
        handler &attr(int attr, int value = 0);

        // run methods
        std::vector<float> run_forward(const std::vector<float> &input);
        std::vector<float> run_gradient(const net::set &set);
        std::vector<float> run_gradient(const std::string &file, bool file_reload);

        // metrics
        signed long get_gradient_performance() const;
        signed long get_forward_performance() const;

        // disk
        void write_net_to_file(const std::string &file);
        void write_set_to_file(const std::string &file, const set &set);
    };
}
#endif