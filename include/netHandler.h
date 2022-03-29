#ifndef NETHANDLER_H
#define NETHANDLER_H

#include <defines.h>
#include <map>
#include <string>
#include <memory>
#include <netFileManager.h>
#include <netAbstract.h>
#include <netGPU.h>

namespace net
{
    class net_abstract;

    constexpr int CPU = 0;
    constexpr int GPU = 1;
#ifdef USE_FPGA
    constexpr int FPGA = 2;
#endif
    constexpr bool RANDOM = true;
    constexpr bool FIXED = false;

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

    public:
        net_handler(const std::string &path);
        ~net_handler();

        void set_active_net(const std::string &net_key);
        void delete_net(const std::string &net_key);
        void net_create_random_from_vector(const std::string &net_key, int implementation, size_t n_ins, const std::vector<size_t> &n_p_l, const std::vector<int> activation_type);
        void net_create(const std::string &net_key, int implementation, bool random, const std::string &file, bool file_reload);
        std::vector<float> active_net_launch_forward(const std::vector<float> &inputs);
        std::vector<float> active_net_launch_gradient(size_t iterations, size_t batch_size,
                                                      float alpha, float alpha_decay, float lambda, float error_threshold, int norm, const std::string &file, bool file_reload);
        std::vector<float> active_net_launch_gradient(const net::net_sets &sets, size_t iterations, size_t batch_size,
                                                      float alpha, float alpha_decay, float lambda, float error_threshold, int norm);
        void active_net_print_inner_vals();
        signed long active_net_get_gradient_performance();
        signed long active_net_get_forward_performance();
        void active_net_write_net_to_file(const std::string &file);
        void process_video(const std::string &video_name);
        std::vector<float> process_img_1000x1000(const std::vector<float> &image);
    };
}
#endif