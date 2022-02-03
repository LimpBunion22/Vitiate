#include <netHandler.h>
#include <netCPU.h>
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

    void net_handler::process_video(const string &video_name)
    {
        // auto it = experimental::filesystem::directory_iterator("./");
        // for (const auto &file : it)
        //     cout << file.path() << endl;

        VideoCapture cap(0); // open the default camera
        if (!cap.isOpened()) // check if we succeeded
        {
            cout << "Fallo al abrir el archivo\n";
            return;
        }

        namedWindow("Camara", WINDOW_AUTOSIZE);
        namedWindow("FPGA", WINDOW_AUTOSIZE);
        // Mat frame;
        image_set in_image = {.resized_image_data=vector<unsigned char>(IMAGE_WIDTH * IMAGE_HEIGHT, 0),
        .original_x_pos = 0,
        .original_y_pos = 0,
        .original_h = 1080,
        .original_w = 1920};


        int batch_load = 0;
        for (;;)
        {
            while (batch_load < 1)
            {
                batch_load++;
                Mat frame;
                cap.read(frame);
                int cn = frame.channels();

                // for (int y = 0; y < min(1080, frame.rows); y++)
                // {
                //     for (int x = 0; x < min(1920, frame.cols); x++)
                //     {
                //         Vec3b &intensity = frame.at<Vec3b>(y, x);
                //         in_image.resized_image_data[y*1920+x] = 0;

                //         for (int k = 0; k < cn; k++)
                //             in_image.resized_image_data[y * 1920 + x] += intensity.val[k];
                //     }
                //     if (min(1920, frame.cols) != 1920)
                //         for (int x = min(1920, frame.cols); x < 1920; x++)
                //             in_image.resized_image_data[y * 1920 + x] = 0;                    
                // }
                // if (min(1080, frame.rows) != 1080)
                //     for (int y = min(1080, frame.cols); y < 1080; y++)
                //         for (int x = 0; x < 1920; x++)
                //             in_image.resized_image_data[y * 1920 + x] = 0;

                for (int x = 0; x < min(1920, frame.cols); x++)
                {
                    for (int y = 0; y < min(1080, frame.rows); y++)
                    {
                        Vec3b &intensity = frame.at<Vec3b>(y, x);
                        in_image.resized_image_data[y + x*1080] = 0;

                        for (int k = 0; k < cn; k++)
                            in_image.resized_image_data[y + x*1080] += intensity.val[k];
                    }
                    if (min(1080, frame.rows) != 1080)
                        for (int y = min(1080, frame.cols); y < 1080; y++)
                                in_image.resized_image_data[y + x*1080] = 0;                                      
                }
                if (min(1920, frame.cols) != 1920)
                    for (int x = min(1920, frame.cols); x < 1920; x++)
                        for (int y = min(1080, frame.cols); y < 1080; y++)
                            in_image.resized_image_data[y + x*1080] = 0;  

                // cout << "Entrando en filter_image\n";
                filter_image(in_image);
                // cout << "Saliendo de filter_image\n";
            }

            Mat frame;
            cap.read(frame);
            image_set out_image = get_filtered_image();
            batch_load--;

            imshow("Camara", frame);
            int cn = frame.channels();
            // for (int y = 0; y < min(1080, frame.rows); y++)
            // {
            //     for (int x = 0; x < min(1920, frame.cols); x++)
            //     {
            //         Vec3b &intensity = frame.at<Vec3b>(y, x);
            //         for(int k = 0; k < cn; k++)
            //             intensity.val[k] = out_image.resized_image_data[y*1920 + x]*6;
            //     }
            // }
            // cout << "pixel 5000: " << out_image.resized_image_data[5000] << "\n";
            // cout << "pixel 500000: " << out_image.resized_image_data[500000] << "\n";
            // cout << "pixel 1000000: " << out_image.resized_image_data[1000000] << "\n";
            for (int x = 0; x < min(1920, frame.cols); x++)
            {
                for (int y = 0; y < min(1080, frame.rows); y++)
                {
                    Vec3b &intensity = frame.at<Vec3b>(y, x);
                    for(int k = 0; k < cn; k++)
                        intensity.val[k] = out_image.resized_image_data[y + x*1080]*2;
                }
            }
            if (waitKey(60) >= 0)
                break;
            imshow("FPGA", frame);
        }
        return;
    }
}
