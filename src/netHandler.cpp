#include <iostream>
#include <netHandler.h>
#include <netCPU.h>
#include <netGPU.h>
#ifdef USE_FPGA
#include <netFPGA.h>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio_c.h>
#endif

namespace net
{
    using namespace std;
#ifdef USE_FPGA
    using namespace cv;
#endif

    void net_handler::set_active_net(const string &net_key)
    {
        if (nets.find(net_key) == nets.end())
            cout << YELLOW << "net " << net_key << " doesn't exist" << RESET << "\n";
        else
        {
            active_net = nets[net_key].get();
            active_net_name = net_key;
        }
    }

    void net_handler::delete_net(const string &net_key)
    {
        if (nets.find(net_key) == nets.end())
            cout << YELLOW << "can't delete nonexistent net " << net_key << RESET << "\n";
        else
        {
            if (active_net_name == net_key)
            {
                active_net = nullptr;
                active_net_name = " ";
            }

            nets.erase(net_key);
        }
    }

    void net_handler::net_create_random_from_vector(const string &net_key, size_t implementation, size_t n_ins, const vector<size_t> &n_p_l, const std::vector<int> activation_type)
    {
        if (nets.find(net_key) != nets.end())
        {
            nets.erase(net_key);
            implementations.erase(net_key);
        }

        switch (implementation)
        {
        case GPU:
            nets[net_key] = unique_ptr<net_abstract>(new gpu::net_gpu(n_ins, n_p_l, activation_type));
            implementations[net_key] = implementation;
            break;
        case CPU:
            nets[net_key] = unique_ptr<net_abstract>(new cpu::net_cpu(n_ins, n_p_l, activation_type));
            implementations[net_key] = implementation;
            break;
#ifdef USE_FPGA
        case FPGA:
            nets[net_key] = unique_ptr<net_abstract>(new fpga::net_fpga());
            implementations[net_key] = implementation;
            break;
#endif
        default:
            cout << RED << "implementation doesn't exist" << RESET << "\n";
            break;
        }
    }

    void net_handler::net_create(const string &net_key, size_t implementation, bool random, const string &file, bool file_reload)
    {
        bool succeeded = random ? manager.load_net_structure(file, file_reload) : manager.load_net(file, file_reload);

        if (!succeeded)
        {
            cout << RED << "failed to create new net " << net_key << " from file \"" << file << '\"' << RESET << "\n";
            return;
        }

        if (nets.find(net_key) != nets.end())
        {
            nets.erase(net_key);
            implementations.erase(net_key);
        }

        switch (implementation)
        {
        case GPU:
            nets[net_key] = unique_ptr<net_abstract>(new gpu::net_gpu(manager.data, random));
            implementations[net_key] = implementation;
            break;
#ifdef USE_FPGA
        case FPGA:
            nets[net_key] = unique_ptr<net_abstract>(new fpga::net_fpga(manager.data, random));
            implementations[net_key] = implementation;
            break;
#endif
        case CPU:
            nets[net_key] = unique_ptr<net_abstract>(new cpu::net_cpu(manager.data, random));
            implementations[net_key] = implementation;
            break;
        default:
            cout << RED << "implementation doesn't exist" << RESET << "\n";
            break;
        }
    }

    vector<float> net_handler::active_net_launch_forward(const vector<float> &inputs)
    {
        if (!active_net)
        {
            cout << YELLOW << "no active net" << RESET << "\n ";
            return vector<float>{-1.0f};
        }
        else
            return active_net->launch_forward(inputs);
    }

    void net_handler::active_net_init_gradient(const string &file, bool file_reload)
    {
        if (!active_net)
            cout << YELLOW << "no active net" << RESET << "\n ";
        else
        {
            bool succeeded = manager.load_sets(file, file_reload);

            if (succeeded)
                active_net->init_gradient(manager.sets);
            else
                cout << RED << "failed to initialize net " << active_net_name << " from file \"" << file << '\"' << RESET "\n";
        }
    }

    vector<float> net_handler::active_net_launch_gradient(int iterations, float error_threshold, float multiplier)
    {
        if (!active_net)
        {
            cout << YELLOW << "no active net" << RESET << "\n ";
            return vector<float>{(float)-1};
        }
        else
            return active_net->launch_gradient(iterations, error_threshold, multiplier);
    }

    void net_handler::active_net_print_inner_vals()
    {
        if (!active_net)
            cout << YELLOW << "no active net" << RESET << "\n ";
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
            cout << YELLOW << "no active net" << RESET << "\n ";
            return -1;
        }
        else
            return active_net->get_gradient_performance();
    }

    signed long net_handler::active_net_get_forward_performance()
    {
        if (!active_net)
        {
            cout << YELLOW << "no active net" << RESET << "\n ";
            return -1;
        }
        else
            return active_net->get_forward_performance();
    }

    void net_handler::active_net_write_net_to_file(const string &file)
    {
        if (!active_net)
            cout << YELLOW << "no active net" << RESET << "\n ";
        else
            manager.write_net_to_file(file, active_net->get_net_data());
    }

    void net_handler::process_video(const string &video_name)
    {
#ifdef USE_FPGA
        if (implementations[active_net_name] != FPGA)
        {
            cout << YELLOW << "active net is not an FPGA implementation" << RESET << "\n";
            return;
        }

        fpga::net_fpga *net = dynamic_cast<fpga::net_fpga *>(active_net);
        // auto it = experimental::filesystem::directory_iterator("./");
        // for (const auto &file : it)
        //     cout << file.path() << endl;

        VideoCapture cap(0); // open the default camera
        if (!cap.isOpened()) // check if we succeeded
        {
            cout << "Fallo al abrir el archivo\n";
            return;
        }
        // cap.set(CAP_PROP_FRAME_WIDTH, 854);
        // cap.set(CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(CAP_PROP_FRAME_WIDTH, 1920);
        cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
        cap.set(CAP_PROP_BUFFERSIZE, 1);

        namedWindow("Camara", WINDOW_NORMAL); // WINDOW_OPENGL
        namedWindow("FPGA", WINDOW_NORMAL);
        // Mat frame;
        unsigned char *red_image = new unsigned char[1920 * 1080]();
        unsigned char *green_image = new unsigned char[1920 * 1080]();
        unsigned char *blue_image = new unsigned char[1920 * 1080]();

        int batch_load = 0;
        Mat cpu_frame;

        cap.read(cpu_frame);
        int cn = cpu_frame.channels();

        for (;;)
        {
            while (batch_load < 1)
            {
                batch_load++;
                cap.read(cpu_frame);

                for (int x = 0; x < min(1920, cpu_frame.cols); x++)
                {
                    for (int y = 0; y < min(1080, cpu_frame.rows); y++)
                    {
                        Vec3b &intensity = cpu_frame.at<Vec3b>(y, x);
                        red_image[y + x * 1080] = (unsigned char)(intensity.val[2]);   // R
                        green_image[y + x * 1080] = (unsigned char)(intensity.val[1]); // G
                        blue_image[y + x * 1080] = (unsigned char)(intensity.val[0]);  // B
                    }
                }

                // cout << "Entrando en filter_image\n";
                net->process_img_1920_1080(red_image, green_image, blue_image);
                // cout << "Saliendo de filter_image\n";
            }

            // cap.read(gpu_frame);
            imshow("Camara", cpu_frame);

            image_set out_image = net->get_img_1920_1080();
            batch_load--;

            for (int x = 0; x < min(1920, cpu_frame.cols); x++)
            {
                for (int y = 0; y < min(1080, cpu_frame.rows); y++)
                {
                    Vec3b &intensity = cpu_frame.at<Vec3b>(y, x);
                    for (int k = 0; k < cn; k++)
                        intensity.val[k] = min(255, out_image.resized_image_data[y + x * 1080] * 3);
                }
            }

            if (waitKey(30) >= 0)
                break;
            imshow("FPGA", cpu_frame);
        }

#else
        cout << YELLOW << "compiled without FPGA suppport" << RESET << "\n";
#endif
    }

    std::vector<float> net_handler::process_img_1000x1000(const vector<float> &image, bool dwz_10)
    {
        // cout << "Llamando al metodo 1000x1000\n";
#ifdef USE_FPGA
        if (implementations[active_net_name] != FPGA)
        {
            cout << YELLOW << "active net is not an FPGA implementation" << RESET << "\n";
            return;
        }

        fpga::net_fpga *net = dynamic_cast<fpga::net_fpga *>(active_net);

        unsigned char *red_image = new unsigned char[1000 * 1000]();
        unsigned char *green_image = new unsigned char[1000 * 1000]();
        unsigned char *blue_image = new unsigned char[1000 * 1000]();

        for (int x = 0; x < 1000 * 1000; x++)
        {
            red_image[x] = (unsigned char)(image[x]);             // R
            green_image[x] = (unsigned char)(image[x + 1000000]); // G
            blue_image[x] = (unsigned char)(image[x + 2000000]);  // B
        }

        // cout << "Enqueuing image\n";
        vector <float>out_image;

        if(dwz_10){
            net->process_img_1000_1000_dwz10(red_image, green_image, blue_image);
            out_image = net->get_img_100_100();
        }
        else{
            net->process_img_1000_1000(red_image, green_image, blue_image);
            out_image = net->get_img_1000_1000();
        }

        // cout << "Returning\n";
        return out_image;
#else
        cout << YELLOW << "compiled without FPGA suppport" << RESET << "\n";
#endif
    }
}
