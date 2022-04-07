#ifndef NETIMAGESTESTER_H
#define NETIMAGESTESTER_H

#include <defines.h>
#include <opencv2/opencv.hpp>

namespace net
{
    constexpr int LEARN_ALL = 0;
    constexpr int LEARN_TRIANGLES = 1;
    constexpr int LEARN_RECTANGLES = 2;
    constexpr int LEARN_ELLIPSES = 3;

    class net_handler;

    class images_tester
    {
    private:
        size_t outs = 0;
        size_t ins = 0;

    private:
        int my_random(int max, int min);
        void ellipse(int w, cv::Mat &img);
        void rectangle(int w, cv::Mat &img);
        void triangle(int w, cv::Mat &img);

    public:
        size_t input_size();
        size_t ouput_size();
        net_sets generate_shapes(int w, int n_images, int type);
        void check_images(net_sets &sets, net_handler &handler, int type);
        void image_demo();
    };
}

#endif