#include <netImagesTester.h>
#include <netHandler.h>

namespace net
{
    using namespace std;
    using namespace cv;

    int images_tester::my_random(int max, int min)
    {
        return (int)((float)random() / (float)RAND_MAX * (max - min) + min);
    }

    void images_tester::ellipse(int w, cv::Mat &img)
    {
        cv::ellipse(img,
                    Point(my_random(w / 2, w / 3), my_random(w / 2, w / 3)),
                    Size(my_random(w / 4, w / 16), my_random(w / 4, w / 16)),
                    my_random(360, 0),
                    0,
                    360,
                    Scalar(255, 255, 255));
    }

    void images_tester::rectangle(int w, cv::Mat &img)
    {
        cv::rectangle(img,
                      Point(my_random(w / 2, 0), my_random(w / 2, 0)),
                      Point(my_random(w, w / 2), my_random(w, w / 2)),
                      Scalar(255, 255, 255));
    }

    void images_tester::triangle(int w, cv::Mat &img)
    {
        Point a = Point(my_random(w / 2, 0), my_random(w / 2, 0));
        Point b = Point(my_random(w, w / 2), my_random(w, w / 2));
        Point c = Point(my_random(w, 0), my_random(w, 0));

        line(img, a, b,
             Scalar(255, 255, 255));
        line(img, a, c,
             Scalar(255, 255, 255));
        line(img, b, c,
             Scalar(255, 255, 255));
    }

    size_t images_tester::input_size()
    {
        return ins;
    }

    size_t images_tester::ouput_size()
    {
        return outs;
    }

    net_set images_tester::generate_shapes(int w, int n_images, int type)
    {
        ins = (size_t)w * w;
        outs = type == LEARN_ALL ? 3 : 1;
        net_set set;
        set.set_ins.reserve(n_images);
        set.set_outs.reserve(n_images);
        set.labels.reserve(n_images);

        for (int i = 0; i < n_images; i++)
        {
            Mat image = Mat::zeros(w, w, CV_8UC1);
            int shape = random();

            if (shape > RAND_MAX / 3 * 2)
            {
                triangle(w, image);

                switch (type)
                {
                case LEARN_TRIANGLES:
                    set.set_outs.emplace_back(vector<float>{1.0f});
                    break;
                case LEARN_ALL:
                    set.set_outs.emplace_back(vector<float>{1.0f, 0.0f, 0.0f});
                    break;
                default:
                    set.set_outs.emplace_back(vector<float>{0.0f});
                    break;
                }

                set.labels.emplace_back(LEARN_TRIANGLES);
            }
            else if (shape > RAND_MAX / 3)
            {
                ellipse(w, image);

                switch (type)
                {
                case LEARN_ELLIPSES:
                    set.set_outs.emplace_back(vector<float>{1.0f});
                    break;
                case LEARN_ALL:
                    set.set_outs.emplace_back(vector<float>{0.0f, 1.0f, 0.0f});
                    break;
                default:
                    set.set_outs.emplace_back(vector<float>{0.0f});
                    break;
                }

                set.labels.emplace_back(LEARN_ELLIPSES);
            }
            else
            {
                rectangle(w, image);

                switch (type)
                {
                case LEARN_RECTANGLES:
                    set.set_outs.emplace_back(vector<float>{1.0f});
                    break;
                case LEARN_ALL:
                    set.set_outs.emplace_back(vector<float>{0.0f, 0.0f, 1.0f});
                    break;
                default:
                    set.set_outs.emplace_back(vector<float>{0.0f});
                    break;
                }

                set.labels.emplace_back(LEARN_RECTANGLES);
            }

            set.set_ins.emplace_back(vector<float>(w * w, 0.0f));
            vector<float> &ref = set.set_ins.back();
            int size = ref.size();

            for (int j = 0; j < size; j++)
                ref[j] = (float)image.data[j];
        }

        cout << "images generator used: " << (set.set_ins.size() * set.set_ins.back().size() + set.set_outs.size() * set.set_outs.back().size()) / 1024 / 1024 * sizeof(float) << " Mbytes\n";
        return set;
    }

    void images_tester::check_images(net_set &set, net_handler &handler, int type)
    {
        if (type == LEARN_ALL)
        {
            int triangles = 0, ellipses = 0, rectangles = 0;
            int correct_triangles = 0, correct_ellipses = 0, correct_rectangles = 0;
            size_t set_size = set.set_ins.size();

            for (size_t i = 0; i < set_size; i++)
            {
                float max = -1.0f;
                int pos = -1, correct_pos = -1;
                auto out = handler.active_net_launch_forward(set.set_ins[i]);
                size_t size = out.size();

                for (int j = 0; j < size; j++)
                {
                    if (out[j] > max)
                    {
                        max = out[j];
                        pos = j;
                    }

                    if (set.set_outs[i][j] != 0.0f)
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

            cout << "identification of every shape: " << correct_ellipses + correct_triangles + correct_rectangles
                 << " shapes of " << triangles + rectangles + ellipses << " correct\n";
            cout << "got " << correct_triangles << " of " << triangles << " triangles\n";
            cout << "got " << correct_rectangles << " of " << rectangles << " rectangles\n";
            cout << "got " << correct_ellipses << " of " << ellipses << " ellipses\n";
        }
        else
        {
            int triangles = 0, ellipses = 0, rectangles = 0;
            int correct_triangles = 0, correct_ellipses = 0, correct_rectangles = 0;
            size_t set_size = set.set_ins.size();

            for (size_t i = 0; i < set_size; i++)
            {
                float max = -1.0f;
                int pos = -1, correct_pos = -1;
                auto out = handler.active_net_launch_forward(set.set_ins[i]);

                switch (set.labels[i])
                {
                case LEARN_TRIANGLES:
                    triangles++;

                    if (out[0] >= 0.5f && set.set_outs[i][0] == 1.0f || out[0] < 0.5f && set.set_outs[i][0] == 0.0f)
                        correct_triangles++;

                    break;
                case LEARN_ELLIPSES:
                    ellipses++;

                    if (out[0] >= 0.5f && set.set_outs[i][0] == 1.0f || out[0] < 0.5f && set.set_outs[i][0] == 0.0f)
                        correct_ellipses++;

                    break;
                case LEARN_RECTANGLES:
                    rectangles++;

                    if (out[0] >= 0.5f && set.set_outs[i][0] == 1.0f || out[0] < 0.5f && set.set_outs[i][0] == 0.0f)
                        correct_rectangles++;

                    break;
                }
            }

            switch (type)
            {
            case LEARN_ELLIPSES:
                cout << "binary classification of ellipses: " << correct_ellipses + correct_triangles + correct_rectangles
                     << " shapes of " << triangles + rectangles + ellipses << " correct\n";
                break;
            case LEARN_RECTANGLES:
                cout << "binary classification of rectangles: " << correct_ellipses + correct_triangles + correct_rectangles
                     << " shapes of " << triangles + rectangles + ellipses << " correct\n";
                break;
            case LEARN_TRIANGLES:
                cout << "binary classification of triangles: " << correct_ellipses + correct_triangles + correct_rectangles
                     << " shapes of " << triangles + rectangles + ellipses << " correct\n";
                break;
            }

            cout << "got " << correct_triangles << " of " << triangles << " triangles\n";
            cout << "got " << correct_rectangles << " of " << rectangles << " rectangles\n";
            cout << "got " << correct_ellipses << " of " << ellipses << " ellipses\n";
        }
    }

    void images_tester::image_demo()
    {
        namedWindow("window", WINDOW_NORMAL);

        while (waitKey() != 27)
        {
            int w = 100;
            Mat image = Mat::zeros(w, w, CV_8UC1);
            int shape = random();

            if (shape > RAND_MAX / 3 * 2)
                triangle(w, image);
            else if (shape > RAND_MAX / 3)
                ellipse(w, image);
            else
                rectangle(w, image);

            imshow("window", image);
        }
    }
}