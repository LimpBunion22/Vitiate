#ifndef NETIMAGESTESTER_H
#define NETIMAGESTESTER_H

#include <defines.h>
#include <opencv2/opencv.hpp>

namespace net
{
    class net_handler;

    class images_tester
    {
    private:
        int w = 100;
        int n_images = 100;
        int triangles = 0;
        int ellipses = 0;
        int rectangles = 0;

    private:
        int my_random(int max, int min);
        void ellipse(cv::Mat &img);
        void rectangle(cv::Mat &img);
        void triangle(cv::Mat &img);

    public:
        void set_attributes(int w, int n_images);
        size_t input_size();
        size_t ouput_size();
        net_sets get_images();
        void check_images(net_sets &sets, net_handler &handler);
    };
}

#endif