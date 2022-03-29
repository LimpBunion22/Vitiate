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

    void images_tester::ellipse(cv::Mat &img)
    {
        cv::ellipse(img,
                    Point(my_random(w / 2, w / 3), my_random(w / 2, w / 3)),
                    Size(my_random(w / 4, w / 16), my_random(w / 4, w / 16)),
                    my_random(360, 0),
                    0,
                    360,
                    Scalar(255, 255, 255));
    }

    void images_tester::rectangle(cv::Mat &img)
    {
        cv::rectangle(img,
                      Point(my_random(w / 2, 0), my_random(w / 2, 0)),
                      Point(my_random(w, w / 2), my_random(w, w / 2)),
                      Scalar(255, 255, 255));
    }

    void images_tester::triangle(cv::Mat &img)
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

    void images_tester::set_attributes(int w, int n_images)
    {
        this->w = w;
        this->n_images = n_images;
    }

    size_t images_tester::input_size()
    {
        return (size_t)w * w;
    }

    size_t images_tester::ouput_size()
    {
        return (size_t)3;
    }

    net_sets images_tester::get_images()
    {
        triangles = rectangles = ellipses = 0;
        net_sets sets;
        sets.set_ins.reserve(n_images);
        sets.set_outs.reserve(n_images);

        for (int i = 0; i < n_images; i++)
        {
            Mat image = Mat::zeros(w, w, CV_8UC1);
            int shape = random();

            if (shape > RAND_MAX / 3 * 2)
            {
                triangle(image);
                sets.set_outs.emplace_back(vector<float>{1.0f, 0.0f, 0.0f});
                triangles++;
            }
            else if (shape > RAND_MAX / 3)
            {
                ellipse(image);
                sets.set_outs.emplace_back(vector<float>{0.0f, 1.0f, 0.0f});
                ellipses++;
            }
            else
            {
                rectangle(image);
                sets.set_outs.emplace_back(vector<float>{0.0f, 0.0f, 1.0f});
                rectangles++;
            }

            sets.set_ins.emplace_back(vector<float>(w * w, 0.0f));
            vector<float> &ref = sets.set_ins.back();
            int size = ref.size();

            for (int j = 0; j < size; j++)
                ref[j] = (float)image.data[j];
        }

        cout << "images generator used: " << (sets.set_ins.size() * sets.set_ins.back().size() + sets.set_outs.size() * sets.set_outs.back().size()) / 1024 / 1024 * sizeof(float) << " Mbytes\n";
        return sets;
    }

    void images_tester::check_images(net_sets &sets, net_handler &handler)
    {
        int correct_triangles = 0, correct_ellipses = 0, correct_rectangles = 0;

        for (int i = 0; i < n_images; i++)
        {
            float max = -1.0f;
            int pos = -1, correct_pos = -1;
            auto out = handler.active_net_launch_forward(sets.set_ins[i]);
            size_t size = out.size();

            for (int j = 0; j < size; j++)
            {
                if (out[j] > max)
                {
                    max = out[j];
                    pos = j;
                }

                if (sets.set_outs[i][j] != 0.0f)
                    correct_pos = j;
            }

            if (pos == correct_pos)
            {
                if (pos == 0)
                    correct_triangles++;
                else if (pos == 1)
                    correct_ellipses++;
                else
                    correct_rectangles++;
            }
        }

        cout << "got " << correct_triangles << " of " << triangles << " triangles\n";
        cout << "got " << correct_rectangles << " of " << rectangles << " rectangles\n";
        cout << "got " << correct_ellipses << " of " << ellipses << " ellipses\n";
    }
}