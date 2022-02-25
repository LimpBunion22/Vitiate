#include <netHandler.h>
#include <netCPU.h>
#include <netGPU.h>
#include <iostream>
#include <netFPGA.h>
#include "opencv2/opencv.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <experimental/filesystem>
#include <opencv2/videoio/videoio_c.h>


namespace net
{
    using namespace std;
    using namespace cv;

    net_handler::~net_handler()
    {
        // gpu::free_resources();
    }

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
        case GPU:
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
        case GPU:
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

                nets[net_key] = unique_ptr<net_abstract>(new gpu::net_gpu(manager.data, derivate, random));
                implementations[net_key] = implementation;
            }
            else
                cout << "failed to create new net " << net_key << " from file \"" << file << "\"\n";
            break;
        case FPGA:
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

                nets[net_key] = unique_ptr<net_abstract>(new fpga::net_fpga(manager.data, derivate, random));
                implementations[net_key] = implementation;
            }
            else
                cout << "failed to create new net " << net_key << " from file \"" << file << "\"\n";
            break;
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

    void net_handler::process_video(const string &video_name)
    {
        if (implementations[active_net_name] != FPGA)
        {
            cout << "active net is not an FPGA implementation\n";
            return;
        }

        fpga::net_fpga *net = (fpga::net_fpga *)active_net;
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

        namedWindow("Camara", WINDOW_NORMAL); //WINDOW_OPENGL
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
                        red_image[y + x * 1080] = (unsigned char)(intensity.val[2]);   //R
                        green_image[y + x * 1080] = (unsigned char)(intensity.val[1]); //G
                        blue_image[y + x * 1080] = (unsigned char)(intensity.val[0]);  //B
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
        return;
    }


    std::vector<unsigned char> net_handler::process_img_1000x1000(const vector<DATA_TYPE> &image)
    {
        cout << "Llamando al metodo 1000x1000\n";
        if (implementations[active_net_name] != FPGA)
        {
            cout << "active net is not an FPGA implementation\n";
            return;
        }

        fpga::net_fpga *net = (fpga::net_fpga *)active_net;
       
        unsigned char *red_image = new unsigned char[1000 * 1000]();
        unsigned char *green_image = new unsigned char[1000 * 1000]();
        unsigned char *blue_image = new unsigned char[1000 * 1000]();

        
        for (int x = 0; x < 1000*1000; x++)
        {
            red_image[x] = (unsigned char)(image[x]);   //R
            green_image[x] = (unsigned char)(image[x + 1000000]); //G
            blue_image[x] = (unsigned char)(image[x + 2000000]);  //B
        }

        cout << "Enqueuing image\n";
        net->process_img_1000_1000(red_image, green_image, blue_image);

        cout << "Reading image\n";
        image_set out_image = net->get_img_1000_1000();
        
        cout << "Returning\n";
        return out_image.resized_image_data;
    }
}
