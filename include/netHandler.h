#ifndef NETHANDLER_H
#define NETHANDLER_H

#include <defines.h>
#include <map>
#include <string>
#include <memory>
#include <netFileManager.h>
#include <netAbstract.h>

namespace net
{
    class net_abstract;

    constexpr size_t CPU = 0;
    constexpr size_t CUDA = 1;
    constexpr size_t FPGA = 2;
    constexpr bool DERIVATE = true;
    constexpr bool NOT_DERIVATE = false;
    constexpr bool RANDOM = true;
    constexpr bool NOT_RANDOM = false;

    class net_handler
    {
    private:
        std::map<std::string, std::unique_ptr<net_abstract>> nets;
        file_manager manager;

    public:
        net_handler(const std::string &path) : manager(path) { srand(time(NULL)); }

        void net_create(const std::string &net_key, size_t implementation, bool derivate, bool random, const std::string &file, bool file_reload = false);
        std::vector<DATA_TYPE> launch_forward(const std::string &net_key, const std::vector<DATA_TYPE> &inputs); //* returns result
        void init_gradient(const std::string &net_key, const std::string &file, bool file_reload = false);
        std::vector<DATA_TYPE> launch_gradient(const std::string &net_key, int iterations); //* returns it times error
        void print_inner_vals(const std::string &net_key);
        signed long get_gradient_performance(const std::string &net_key);
        signed long get_forward_performance(const std::string &net_key);
        void write_net_to_file(const std::string &net_key, const std::string &file);
    };
}
#endif