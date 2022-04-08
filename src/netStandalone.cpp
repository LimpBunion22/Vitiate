// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <pybind11/stl_bind.h>
// #include <netHandler.h>

// namespace py = pybind11;
// using namespace std;
// PYBIND11_MAKE_OPAQUE(vector<float>);
// PYBIND11_MAKE_OPAQUE(vector<size_t>);
// PYBIND11_MAKE_OPAQUE(vector<unsigned char>);
// PYBIND11_MAKE_OPAQUE(vector<int>);

// PYBIND11_MODULE(netStandalone, m)
// {
//     m.doc() = "Vitiate AI library for floats";

//     py::bind_vector<vector<float>>(m, "v_float");
//     py::bind_vector<vector<size_t>>(m, "v_size_t");
//     py::bind_vector<vector<unsigned char>>(m, "v_uchar");
//     py::bind_vector<vector<int>>(m, "v_int");

//     //*attributes
//     m.attr("ALPHA") = py::int_(net::ALPHA);
//     m.attr("ALPHA_DECAY") = py::int_(net::ALPHA_DECAY);
//     m.attr("ERROR_THRESHOLD") = py::int_(net::ERROR_THRESHOLD);
//     m.attr("REG_LAMBDA") = py::int_(net::REG_LAMBDA);
//     m.attr("NORM") = py::int_(net::NORM);
//     m.attr("DROPOUT_INTERVAL") = py::int_(net::DROPOUT_INTERVAL);
//     m.attr("ADAM") = py::int_(net::ADAM);
//     m.attr("MOMENTUM_BETA") = py::int_(net::MOMENTUM_BETA);
//     m.attr("RMS_BETA") = py::int_(net::RMS_BETA);

//     //*selections
//     m.attr("FULL_BATCH") = py::size_t(net::FULL_BATCH);
//     m.attr("ON") = py::int_(net::ON);
//     m.attr("OFF") = py::int_(net::OFF);

//     //*normalization
//     m.attr("NO_NORM_REG") = py::int_(net::NO_NORM_REG);
//     m.attr("REG") = py::int_(net::REG);
//     m.attr("NORM_0") = py::int_(net::NORM_0);
//     m.attr("NORM_1") = py::int_(net::NORM_1);
//     m.attr("NORM_2") = py::int_(net::NORM_2);
//     m.attr("NORM_REG_0") = py::int_(net::NORM_REG_0);
//     m.attr("NORM_REG_1") = py::int_(net::NORM_REG_1);
//     m.attr("NORM_REG_2") = py::int_(net::NORM_REG_2);

//     //*activations
//     m.attr("RELU") = py::int_(net::RELU);
//     m.attr("RELU2") = py::int_(net::RELU2);
//     m.attr("SIGMOID") = py::int_(net::SIGMOID);
//     m.attr("RELU2_SOFT_MAX") = py::int_(net::RELU2_SOFT_MAX);

//     //*implementation
//     m.attr("CPU") = py::int_(net::CPU);
//     m.attr("GPU") = py::int_(net::GPU);
// #ifdef USE_FPGA
//     m.attr("FPGA") = py::int(net::FPGA);
// #endif
//     m.attr("RANDOM_NET") = py::bool_(net::RANDOM_NET);
//     m.attr("FIXED_NET") = py::bool_(net::FIXED_NET);
//     m.attr("RELOAD_FILE") = py::bool_(net::RELOAD_FILE);
//     m.attr("REUSE_FILE") = py::bool_(net::REUSE_FILE);

//     py::class_<net::image_set>(m, "image_set")
//         .def(py::init<>())
//         .def_readwrite("resized_image_data", &net::image_set::resized_image_data)
//         .def_readwrite("original_x_pos", &net::image_set::original_x_pos)
//         .def_readwrite("original_y_pos", &net::image_set::original_y_pos)
//         .def_readwrite("original_h", &net::image_set::original_h)
//         .def_readwrite("original_w", &net::image_set::original_w);

//     py::class_<net::net_handler>(m, "net_handler")
//         .def(py::init<const string &>(), py::arg("path"))
//         .def("set_active_net", &net::net_handler::set_active_net, py::arg("net_key"))
//         .def("delete_net", &net::net_handler::delete_net, py::arg("net_key"))
//         .def("net_create_random_from_vector", &net::net_handler::net_create_random_from_vector,
//              py::arg("net_key"), py::arg("implementation"), py::arg("n_ins"), py::arg("n_p_l"), py::arg("activation_type"))
//         .def("net_create", &net::net_handler::net_create, py::arg("net_key"), py::arg("implementation"),
//              py::arg("random"), py::arg("file"), py::arg("file_reload"))
//         .def("active_net_launch_forward", &net::net_handler::active_net_launch_forward, py::arg("inputs"))
//         .def("active_net_set_gradient_attribute", &net::net_handler::active_net_set_gradient_attribute, py::arg("attribute"), py::arg("value"))
//         .def("active_net_launch_gradient", py::overload_cast<size_t, size_t, const string &, bool>(&net::net_handler::active_net_launch_gradient),
//              py::arg("iterations"), py::arg("batch_size"), py::arg("file"), py::arg("file_reload"))
//         .def("active_net_print_inner_vals", &net::net_handler::active_net_print_inner_vals)
//         .def("active_net_get_gradient_performance", &net::net_handler::active_net_get_gradient_performance)
//         .def("active_net_get_forward_performance", &net::net_handler::active_net_get_forward_performance)
//         .def("active_net_write_to_file", &net::net_handler::active_net_write_to_file, py::arg("file"))
//         .def("process_video", &net::net_handler::process_video, py::arg("video_name"))
//         .def("process_img_1000x1000", &net::net_handler::process_img_1000x1000, py::arg("image"), py::arg("dwz_10") = false);
// }

#include <defines.h>
#include <netHandler.h>
#include <netImagesTester.h>

using namespace std;

int main()
{
    net::net_handler handler("/home/gabi/workspace_development");
    float alpha = 30.0f;
    float alpha_decay = 0.00001f;
    float error_threshold = 0.00001f;
    float lambda = 0.1f;
    int n_nets = 8;

    net::images_tester images;
    net::net_set set = images.generate_shapes(100, 300, net::LEARN_ALL);
    handler.normalize_image_set(set, net::MIN_MAX, net::PER_IMAGE, 1.0f, 0.0f);

    size_t n_ins = images.input_size();
    size_t n_outs = images.ouput_size();
    vector<size_t> n_p_l = {15, 15, n_outs};
    vector<int> activation_type = {
        net::RELU2,
        net::RELU2,
        net::RELU2_SOFT_MAX};
    handler.net_create_random_from_vector("gpu", net::CPU, n_ins, n_p_l, activation_type);
    handler.set_active_net("gpu");
    handler.active_net_set_gradient_attribute(net::ALPHA, alpha);
    handler.active_net_set_gradient_attribute(net::ALPHA_DECAY, alpha_decay);
    handler.active_net_set_gradient_attribute(net::REG_LAMBDA, lambda);
    handler.active_net_set_gradient_attribute(net::ERROR_THRESHOLD, error_threshold);
    handler.active_net_set_gradient_attribute(net::NORM, net::NORM_1);
    handler.active_net_set_gradient_attribute(net::ADAM, net::ON);
    auto out = handler.active_net_launch_gradient(set, 50, 64);

    for (auto &i : out)
        cout << i << " ";

    cout << "\n";
    cout << handler.active_net_get_gradient_performance() << "\n";

    cout << "training set\n";
    images.check_images(set, handler, net::LEARN_ALL);
    cout << "validation set\n";
    net::net_set validation = images.generate_shapes(100, 50, net::LEARN_ALL);
    handler.normalize_image_set(validation, net::MIN_MAX, net::PER_IMAGE, 1.0f, 0.0f);
    images.check_images(validation, handler, net::LEARN_ALL);

    // for (int i = 0; i < n_nets; i++)
    // {
    //     handler.net_create("my_net", net::GPU, net::FIXED, "base_net", net::REUSE_FILE);
    //     handler.set_active_net("my_net");
    //     cout << "alpha: " << alpha << ", alpha decay: " << alpha_decay << ", lambda: " << lambda << "\n";

    //     int norm = -1;
    //     switch (i)
    //     {
    //     case 0:
    //         norm = net::NO_NORM_REG;
    //         cout << "no norm reg\n";
    //         break;
    //     case 1:
    //         norm = net::REG;
    //         cout << "reg\n";
    //         break;
    //     case 2:
    //         norm = net::NORM_0;
    //         cout << "norm 0\n";
    //         break;
    //     case 3:
    //         norm = net::NORM_1;
    //         cout << "norm 1\n";
    //         break;
    //     case 4:
    //         norm = net::NORM_2;
    //         cout << "norm 2\n";
    //         break;
    //     case 5:
    //         norm = net::NORM_REG_0;
    //         cout << "norm reg 0\n";
    //         break;
    //     case 6:
    //         norm = net::NORM_REG_1;
    //         cout << "norm reg 1\n";
    //         break;
    //     case 7:
    //         norm = net::NORM_REG_2;
    //         cout << "norm reg 2\n";
    //         break;
    //     }

    //     auto out = handler.active_net_launch_gradient(4, 32, alpha, alpha_decay, lambda, error_threshold, norm, 0, "base_set", net::REUSE_FILE);
    //     size_t size = out.size();

    //     for (size_t i = 0; i < size; i++)
    //         if (i % 10 == 0)
    //             cout << YELLOW << out[i] << " ";
    //         else
    //             cout << RESET << out[i] << " ";

    //     cout << "\n";
    //     cout << handler.active_net_get_gradient_performance() << "\n";
    //     handler.delete_net("my_net");
    // }

    return 0;
}