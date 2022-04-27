#ifndef SHAPEGENERATOR_H
#define SHAPEGENERATOR_H

#include <defines.h>
#include <opencv2/opencv.hpp>

namespace net
{
    constexpr int LEARN_ALL = 0;
    constexpr int LEARN_TRIANGLES = 1;
    constexpr int LEARN_RECTANGLES = 2;
    constexpr int LEARN_ELLIPSES = 3;

    class handler;

    class shape_generator
    {
    private:
        int _out_size = 0;
        int _in_size = 0;

    private:
        int my_random(int max, int min);
        void ellipse(int w, cv::Mat &img);
        void rectangle(int w, cv::Mat &img);
        void triangle(int w, cv::Mat &img);

    public:
        int input_size();
        int output_size();
        set generate_shapes(int w, int n_images, int type);
        void check_shapes(set &set, handler &handler, int type);
        void image_demo();
    };
}

#endif