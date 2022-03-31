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
//     m.attr("RELU2") = py::int_(net::RELU2);
//     m.attr("SIGMOID") = py::int_(net::SIGMOID);
//     m.attr("RELU2_SOFT_MAX") = py::int_(net::RELU2_SOFT_MAX);
//     m.attr("FULL_BATCH") = py::size_t(net::FULL_BATCH);
//     m.attr("NO_NORM_REG") = py::int_(net::NO_NORM_REG);
//     m.attr("REG_ONLY") = py::int_(net::REG_ONLY);
//     m.attr("NORM_0") = py::int_(net::NORM_0);
//     m.attr("NORM_1") = py::int_(net::NORM_1);
//     m.attr("NORM_2") = py::int_(net::NORM_2);
//     m.attr("NORM_REG_0") = py::int_(net::NORM_REG_0);
//     m.attr("NORM_REG_1") = py::int_(net::NORM_REG_1);
//     m.attr("NORM_REG_2") = py::int_(net::NORM_REG_2);
//     m.attr("CPU") = py::int_(net::CPU);
//     m.attr("GPU") = py::int_(net::GPU);
// #ifdef USE_FPGA
//     m.attr("FPGA") = py::int(net::FPGA);
// #endif
//     m.attr("RANDOM") = py::bool_(net::RANDOM);
//     m.attr("FIXED") = py::bool_(net::FIXED);
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
//         .def("active_net_launch_gradient", &net::net_handler::active_net_launch_gradient, py::arg("iterations"),
//              py::arg("batch_size"), py::arg("alpha"), py::arg("alpha_decay"), py::arg("reg_lambda"),
//              py::arg("error_threshold"), py::arg("norm"), py::arg("file"), py::arg("file_reload"))
//         .def("active_net_print_inner_vals", &net::net_handler::active_net_print_inner_vals)
//         .def("active_net_get_gradient_performance", &net::net_handler::active_net_get_gradient_performance)
//         .def("active_net_get_forward_performance", &net::net_handler::active_net_get_forward_performance)
//         .def("active_net_write_net_to_file", &net::net_handler::active_net_write_net_to_file, py::arg("file"))
//         .def("process_video", &net::net_handler::process_video, py::arg("video_name"))
//         .def("process_img_1000x1000", &net::net_handler::process_img_1000x1000, py::arg("image"), py::arg("dwz_10") = false);
// }

#include <defines.h>
#include <netHandler.h>
#include <netImagesTester.h>

using namespace std;

int main()
{
    float alpha = 10.0f;
    float alpha_decay = 0.00001f;
    float error_threshold = 0.00001f;
    float lambda = 0.1f;
    int n_nets = 4;

    net::net_handler handler("/home/gabi/workspace_development");
    net::images_tester images;
    net::net_sets sets = images.generate_shapes(100, 300, net::LEARN_RECTANGLES);
    size_t n_ins = images.input_size();
    size_t n_outs = images.ouput_size();
    vector<size_t> n_p_l = {50, 50, 6, n_outs};
    vector<int> activation_type = {
        net::RELU2,
        net::RELU2,
        net::RELU2,
        net::SIGMOID};
    handler.net_create_random_from_vector("gpu", net::GPU, n_ins, n_p_l, activation_type);
    handler.set_active_net("gpu");
    auto out = handler.active_net_launch_gradient(sets, 400, net::FULL_BATCH, alpha, alpha_decay, lambda, error_threshold, net::NORM_1);
    size_t size = out.size();

    for (size_t i = 0; i < size; i++)
        if (i % 10 == 0)
            cout << YELLOW << out[i] << " ";
        else
            cout << RESET << out[i] << " ";

    cout << "\n";
    cout << handler.active_net_get_gradient_performance() << "\n";
    images.check_images(sets, handler, net::LEARN_RECTANGLES);

    // for (int i = 0; i < n_nets; i++)
    // {
    //     handler.net_create("my_net", net::GPU, net::FIXED, "base_net", net::REUSE_FILE);
    //     handler.set_active_net("my_net");
    //     cout << "alpha: " << alpha << ", alpha decay: " << alpha_decay << "\n";
    //     auto out = handler.active_net_launch_gradient(4, 32, alpha, alpha_decay, lambda, error_threshold, net::NORM_1, "base_sets", net::REUSE_FILE);
    //     size_t size = out.size();

    //     for (size_t i = 0; i < size; i++)
    //         if (i % 10 == 0)
    //             cout << YELLOW << out[i] << " ";
    //         else
    //             cout << RESET << out[i] << " ";

    //     cout << "\n";
    //     cout << handler.active_net_get_gradient_performance() << "\n";
    //     handler.delete_net("my_net");
    //     alpha *= 10;
    // }

    return 0;
}