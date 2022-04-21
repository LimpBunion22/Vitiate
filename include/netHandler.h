#ifndef NETHANDLER_H
#define NETHANDLER_H

#include <defines.h>
#include <map>
#include <string>
#include <memory>
#include <netFileManager.h>
#include <netAbstract.h>
#include <netGPU.h>
#include <fpgaHandler.h>

namespace net
{
    class net_abstract;

    constexpr int CPU = 0;
    constexpr int GPU = 1;
#ifdef USE_FPGA
    constexpr int FPGA = 2;
#endif
    constexpr bool RANDOM_NET = true;
    constexpr bool FIXED_NET = false;

    class net_handler
    {
    private:
        std::map<std::string, std::unique_ptr<net_abstract>> nets;
        std::map<std::string, int> implementations;
        file_manager manager;
        net_abstract *active_net;
        std::string active_net_name;
        cudaStream_t stream;
        gpu::cuda_libs_data libs_data;

        fpga::fpga_handler mustang_handler;
        bool mustang_handler_init = false;

    public:
        net_handler(const std::string &path);
        ~net_handler();

        void set_active_net(const std::string &net_key);
        void delete_net(const std::string &net_key);
        void net_create_random_from_vector(const std::string &net_key, int implementation, size_t n_ins, const std::vector<size_t> &n_p_l, const std::vector<int> activation_type);
        void net_create(const std::string &net_key, int implementation, bool random, const std::string &file, bool file_reload);
        std::vector<float> active_net_launch_forward(const std::vector<float> &inputs);
        void active_net_set_gradient_attribute(int attribute, float value);
        std::vector<float> active_net_launch_gradient(size_t iterations, size_t batch_size, const std::string &file, bool file_reload);
        std::vector<float> active_net_launch_gradient(const net::net_set &set, size_t iterations, size_t batch_size);
        void active_net_print_inner_vals();
        signed long active_net_get_gradient_performance();
        signed long active_net_get_forward_performance();
        void active_net_write_to_file(const std::string &file);
        void write_sets_to_file(const std::string &file, const net_sets &sets);
        // void process_video(const std::string &video_name);
        // std::vector<float> process_img_1000x1000(const std::vector<float> &image, bool dwz_10 = false);
        void enq_fpga_net(const std::string &net_key, const std::vector<float> &inputs, bool reload=true, bool same_in=false, bool big_nets=false);
        void exe_fpga_nets();
        std::vector<float> read_fpga_net(const std::string &net_key);
    };
}
#endif