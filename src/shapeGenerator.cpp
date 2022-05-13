#include <shapeGenerator.h>
#include <netHandler.h>
#include <utils.h>

namespace net
{
    int shape_generator::my_random(int max, int min)
    {
        auto &rand = rand_generator::get_instance();
        auto dis = rand.int_distribution(min, max);
        auto &gen = rand.generator();
        return dis(gen);
    }

    void shape_generator::ellipse(int w, cv::Mat &img)
    {
        cv::ellipse(img,
                    cv::Point(my_random(w / 2, w / 3), my_random(w / 2, w / 3)),
                    cv::Size(my_random(w / 4, w / 16), my_random(w / 4, w / 16)),
                    my_random(360, 0),
                    0,
                    360,
                    cv::Scalar(255, 255, 255));
    }

    void shape_generator::rectangle(int w, cv::Mat &img)
    {
        cv::rectangle(img,
                      cv::Point(my_random(w / 2, 0), my_random(w / 2, 0)),
                      cv::Point(my_random(w, w / 2), my_random(w, w / 2)),
                      cv::Scalar(255, 255, 255));
    }

    void shape_generator::triangle(int w, cv::Mat &img)
    {
        cv::Point a = cv::Point(my_random(w / 2, 0), my_random(w / 2, 0));
        cv::Point b = cv::Point(my_random(w, w / 2), my_random(w, w / 2));
        cv::Point c = cv::Point(my_random(w, 0), my_random(w, 0));

        line(img, a, b,
             cv::Scalar(255, 255, 255));
        line(img, a, c,
             cv::Scalar(255, 255, 255));
        line(img, b, c,
             cv::Scalar(255, 255, 255));
    }

    int shape_generator::input_size()
    {
        return _in_size;
    }

    int shape_generator::output_size()
    {
        return _out_size;
    }

    set shape_generator::generate_shapes(int w, int n_images, int type)
    {
        _in_size = w * w;
        _out_size = type == LEARN_ALL ? 3 : 1;
        set set;
        set.data_num = n_images;
        set.input_data.reserve(n_images * _in_size);
        set.output_data.reserve(n_images * _out_size);
        set.labels.reserve(n_images);

        for (int i = 0; i < n_images; i++)
        {
            cv::Mat image = cv::Mat::zeros(w, w, CV_8UC1);
            int shape = my_random(2, 0);

            switch (shape)
            {
            case 2:
            {
                triangle(w, image);

                switch (type)
                {
                case LEARN_TRIANGLES:
                    set.output_data.emplace_back(1.0f);
                    break;
                case LEARN_ALL:
                    set.output_data.emplace_back(1.0f);
                    set.output_data.emplace_back(0.0f);
                    set.output_data.emplace_back(0.0f);
                    break;
                default:
                    set.output_data.emplace_back(0.0f);
                    break;
                }

                set.labels.emplace_back(LEARN_TRIANGLES);
                break;
            }

            case 1:
            {
                ellipse(w, image);

                switch (type)
                {
                case LEARN_ELLIPSES:
                    set.output_data.emplace_back(1.0f);
                    break;
                case LEARN_ALL:
                    set.output_data.emplace_back(0.0f);
                    set.output_data.emplace_back(1.0f);
                    set.output_data.emplace_back(0.0f);
                    break;
                default:
                    set.output_data.emplace_back(0.0f);
                    break;
                }

                set.labels.emplace_back(LEARN_ELLIPSES);
                break;
            }

            case 0:
            default:
            {
                rectangle(w, image);

                switch (type)
                {
                case LEARN_RECTANGLES:
                    set.output_data.emplace_back(1.0f);
                    break;
                case LEARN_ALL:
                    set.output_data.emplace_back(0.0f);
                    set.output_data.emplace_back(0.0f);
                    set.output_data.emplace_back(1.0f);
                    break;
                default:
                    set.output_data.emplace_back(0.0f);
                    break;
                }

                set.labels.emplace_back(LEARN_RECTANGLES);
                break;
            }
            }

            for (int j = 0; j < _in_size; j++) // store image
                set.input_data.emplace_back((float)image.data[j]);
        }

        std::cout << "images generator used: " << (set.input_data.size() + set.output_data.size() + set.labels.size()) / 1024 / 1024 * sizeof(float) << " Mbytes\n";
        return set;
    }

    void shape_generator::check_shapes(set &set, handler &handler, int type)
    {
        if (type == LEARN_ALL)
        {
            int triangles = 0, ellipses = 0, rectangles = 0;
            int correct_triangles = 0, correct_ellipses = 0, correct_rectangles = 0;
            int input_size = (int)set.input_data.size() / set.data_num;
            int output_size = (int)set.output_data.size() / set.data_num;
            std::vector<float> input(input_size);

            for (int i = 0; i < set.data_num; i++)
            {
                float max = std::numeric_limits<float>::min();
                int pos = -1, correct_pos = -1;
                memcpy(input.data(), set.input_data.data() + i * input_size, input_size * sizeof(float)); // get input data
                auto out = handler.run_forward(input);
                int size = out.size();

                for (int j = 0; j < size; j++)
                {
                    if (out[j] > max)
                    {
                        max = out[j];
                        pos = j;
                    }

                    if (set.output_data[i * output_size + j] != 0.0f)
                        correct_pos = j;
                }

                switch (set.labels[i])
                {
                case LEARN_TRIANGLES:
                    triangles++;

                    if (pos == correct_pos && pos == 0)
                        correct_triangles++;

                    break;
                case LEARN_ELLIPSES:
                    ellipses++;

                    if (pos == correct_pos && pos == 1)
                        correct_ellipses++;

                    break;
                case LEARN_RECTANGLES:
                    rectangles++;

                    if (pos == correct_pos && pos == 2)
                        correct_rectangles++;

                    break;
                }
            }

            std::cout << "identification of every shape: " << correct_ellipses + correct_triangles + correct_rectangles
                      << " shapes of " << triangles + rectangles + ellipses << " correct\n";
            std::cout << "got " << correct_triangles << " of " << triangles << " triangles\n";
            std::cout << "got " << correct_rectangles << " of " << rectangles << " rectangles\n";
            std::cout << "got " << correct_ellipses << " of " << ellipses << " ellipses\n";
        }
        else
        {
            int triangles = 0, ellipses = 0, rectangles = 0;
            int correct_triangles = 0, correct_ellipses = 0, correct_rectangles = 0;
            int input_size = (int)set.input_data.size() / set.data_num;
            int output_size = (int)set.output_data.size() / set.data_num;
            std::vector<float> input(input_size);

            for (int i = 0; i < set.data_num; i++)
            {
                float max = std::numeric_limits<float>::min();
                int pos = -1, correct_pos = -1;
                memcpy(input.data(), set.input_data.data() + i * input_size, input_size * sizeof(float)); // get input data
                auto out = handler.run_forward(input);

                switch (set.labels[i])
                {
                case LEARN_TRIANGLES:
                    triangles++;

                    if (out[0] >= 0.5f && set.output_data[i * output_size] == 1.0f || out[0] < 0.5f && set.output_data[i * output_size] == 0.0f)
                        correct_triangles++;

                    break;
                case LEARN_ELLIPSES:
                    ellipses++;

                    if (out[0] >= 0.5f && set.output_data[i * output_size] == 1.0f || out[0] < 0.5f && set.output_data[i * output_size] == 0.0f)
                        correct_ellipses++;

                    break;
                case LEARN_RECTANGLES:
                    rectangles++;

                    if (out[0] >= 0.5f && set.output_data[i * output_size] == 1.0f || out[0] < 0.5f && set.output_data[i * output_size] == 0.0f)
                        correct_rectangles++;

                    break;
                }
            }

            switch (type)
            {
            case LEARN_ELLIPSES:
                std::cout << "binary classification of ellipses: " << correct_ellipses + correct_triangles + correct_rectangles
                          << " shapes of " << triangles + rectangles + ellipses << " correct\n";
                break;
            case LEARN_RECTANGLES:
                std::cout << "binary classification of rectangles: " << correct_ellipses + correct_triangles + correct_rectangles
                          << " shapes of " << triangles + rectangles + ellipses << " correct\n";
                break;
            case LEARN_TRIANGLES:
                std::cout << "binary classification of triangles: " << correct_ellipses + correct_triangles + correct_rectangles
                          << " shapes of " << triangles + rectangles + ellipses << " correct\n";
                break;
            }

            std::cout << "got " << correct_triangles << " of " << triangles << " triangles\n";
            std::cout << "got " << correct_rectangles << " of " << rectangles << " rectangles\n";
            std::cout << "got " << correct_ellipses << " of " << ellipses << " ellipses\n";
        }
    }

    void shape_generator::image_demo()
    {
        cv::namedWindow("window", cv::WINDOW_NORMAL);

        while (cv::waitKey() != 27)
        {
            int w = 100;
            cv::Mat image = cv::Mat::zeros(w, w, CV_8UC1);
            int shape = my_random(2, 0);

            switch (shape)
            {
            case 2:
                triangle(w, image);
                break;

            case 1:
                ellipse(w, image);
                break;

            case 0:
            default:
                rectangle(w, image);
                break;
            }

            imshow("window", image);
        }
    }
}