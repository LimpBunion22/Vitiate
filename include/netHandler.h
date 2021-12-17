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
    constexpr size_t MULTI = 3;
    constexpr bool DERIVATE = true;
    constexpr bool NOT_DERIVATE = false;
    constexpr bool RANDOM = true;
    constexpr bool NOT_RANDOM = false;

    class net_handler
    {
    private:
        std::map<std::string, std::unique_ptr<net_abstract>> nets;
        std::map<std::string, size_t> implementations;
        file_manager manager;
        net_abstract *active_net;
        std::string active_net_name;

    public:
        net_handler(const std::string &path) : manager(path), active_net(nullptr) { srand(time(NULL)); }
        ~net_handler()
        { // TODO implement multi::free_pool();
        }

        void set_active_net(const std::string &net_key);
        void net_create(const std::string &net_key, size_t implementation, bool derivate, bool random, const std::string &file, bool file_reload = false);
        std::vector<DATA_TYPE> active_net_launch_forward(const std::vector<DATA_TYPE> &inputs); //* returns result
        void active_net_init_gradient(const std::string &file, bool file_reload = false);
        std::vector<DATA_TYPE> active_net_launch_gradient(int iterations); //* returns it times error
        void active_net_print_inner_vals();
        signed long active_net_get_gradient_performance();
        signed long active_net_get_forward_performance();
        void active_net_write_net_to_file(const std::string &file);
    };
}
#endif