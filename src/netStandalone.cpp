#ifdef BIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <netHandler.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned char>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);

PYBIND11_MODULE(netStandalone, m)
{
     m.doc() = "Vitiate AI library for floats";

     py::bind_vector<std::vector<float>>(m, "v_float");
     py::bind_vector<std::vector<unsigned char>>(m, "v_uchar");
     py::bind_vector<std::vector<int>>(m, "v_int");

     m.attr("DISABLE") = py::int_(net::DISABLE);

     //*basic attributes
     m.attr("ALPHA") = py::int_(net::ALPHA);
     m.attr("ALPHA_DECAY") = py::int_(net::ALPHA_DECAY);
     m.attr("ERROR_THRESHOLD") = py::int_(net::ERROR_THRESHOLD);
     m.attr("EPOCHS") = py::int_(net::EPOCHS);
     m.attr("BATCH_SIZE") = py::int_(net::BATCH_SIZE);

     // algos
     m.attr("ADAM") = py::int_(net::ADAM);
     m.attr("MOMENTUM_BETA") = py::int_(net::MOMENTUM_BETA);
     m.attr("RMS_BETA") = py::int_(net::RMS_BETA);
     m.attr("REG") = py::int_(net::REG);
     m.attr("REG_LAMBDA") = py::int_(net::REG_LAMBDA);
     m.attr("DROPOUT") = py::int_(net::DROPOUT);

     //*normalization
     m.attr("MAX") = py::int_(net::MAX);
     m.attr("ABS") = py::int_(net::ABS);
     m.attr("MODULO") = py::int_(net::MODULO);
     m.attr("MIN_MAX") = py::int_(net::MIN_MAX);
     m.attr("STANDARIZATION") = py::int_(net::STANDARIZATION);

     //*activations
     m.attr("RELU") = py::int_(net::RELU);
     m.attr("RELU2") = py::int_(net::RELU2);
     m.attr("SIGMOID") = py::int_(net::SIGMOID);
     m.attr("RELU2_SOFT_MAX") = py::int_(net::RELU2_SOFT_MAX);

     //*implementations
     m.attr("CPU") = py::int_(net::CPU);
     m.attr("GPU") = py::int_(net::GPU);
#ifdef USE_FPGA
     m.attr("FPGA") = py::int_(net::FPGA);
#endif

     // others
     m.attr("RELOAD_FILE") = py::bool_(net::RELOAD_FILE);
     m.attr("REUSE_FILE") = py::bool_(net::REUSE_FILE);

     py::class_<net::image_set>(m, "image_set")
         .def(py::init<>())
         .def_readwrite("resized_image_data", &net::image_set::resized_image_data)
         .def_readwrite("original_x_pos", &net::image_set::original_x_pos)
         .def_readwrite("original_y_pos", &net::image_set::original_y_pos)
         .def_readwrite("original_h", &net::image_set::original_h)
         .def_readwrite("original_w", &net::image_set::original_w);

     py::class_<net::handler>(m, "handler")
         // management
         .def(py::init<const std::string &>(), py::arg("path"))
         .def("clone", &net::handler::clone, py::arg("original"), py::arg("clone"))
         .def("set_active_net", &net::handler::set_active_net)
         .def("delete_net", &net::handler::delete_net)
         .def("instantiate", &net::handler::instantiate, py::arg("key"), py::arg("implementation"))
         .def("set_input_size", &net::handler::set_input_size)
         .def("build_fully_layer", &net::handler::build_fully_layer, py::arg("layer_size"), py::arg("activation") = net::RELU2)
         .def("build_net", &net::handler::build_net)
         .def("build_net_from_file", &net::handler::build_net_from_file, py::arg("file"), py::arg("file_reload"))
         .def("build_net_from_data", &net::handler::build_net_from_data, py::arg("input_size"), py::arg("n_p_l"), py::arg("activations"))
         .def("attr", py::overload_cast<int, float>(&net::handler::attr), py::arg("attr"), py::arg("value"),
              py::return_value_policy::reference)
         .def("attr", py::overload_cast<int, int>(&net::handler::attr), py::arg("attr"), py::arg("value") = 0,
              py::return_value_policy::reference)

         // run methods
         .def("run_forward", &net::handler::run_forward)
         .def("run_gradient", py::overload_cast<const std::string &, bool>(&net::handler::run_gradient),
              py::arg("file"), py::arg("file_reload"))
         .def("mutate", &net::handler::mutate)

         // metrics
         .def("get_gradient_performance", &net::handler::get_gradient_performance)
         .def("get_forward_performance", &net::handler::get_forward_performance)

         // disk
         .def("write_net_to_file", &net::handler::write_net_to_file)

// fpga
#ifdef USE_FPGA
         .def("enq_fpga_net", &net::net_handler::enq_fpga_net, py::arg("net_key"), py::arg("inputs"), py::arg("reload") = true, py::arg("same_in") = false, py::arg("big_nets") = false)
         .def("exe_fpga_nets", &net::net_handler::exe_fpga_nets)
         .def("read_fpga_net", &net::net_handler::read_fpga_net, py::arg("net_key"));
#else
         ; // close semicolon
#endif
}

#else
#include <defines.h>
#include <netHandler.h>
#include <shapeGenerator.h>

int main()
{
     net::handler handler("/home/gabi/workspace_development");
     float alpha = 50.0f;
     float alpha_decay = 0.00001f;
     float error_threshold = 0.00001f;
     float lambda = 0.1f;

     net::shape_generator shapes;
     net::set set = shapes.generate_shapes(100, 300, net::LEARN_ALL);

     handler.instantiate("net0", net::GPU);
     handler.set_active_net("net0");
     handler.set_input_size(shapes.input_size());
     handler.build_fully_layer(15);
     handler.build_fully_layer(15);
     handler.build_fully_layer(shapes.output_size(), net::RELU2_SOFT_MAX);
     handler.build_net();
     // handler.build_net_from_file("net", net::REUSE_FILE);

     handler.attr(net::EPOCHS, 50)
         .attr(net::BATCH_SIZE, 64)
         .attr(net::ALPHA, alpha)
         .attr(net::ALPHA_DECAY, alpha_decay)
         .attr(net::ERROR_THRESHOLD, error_threshold)
         .attr(net::ABS)
         .attr(net::ADAM);

     // auto out = handler.run_gradient("set", net::REUSE_FILE);
     auto out = handler.run_gradient(set);

     for (auto &i : out)
          std::cout << i << " ";

     std::cout << "\n";
     std::cout << handler.get_gradient_performance() << "\n";

     std::cout << "training set\n";
     shapes.check_shapes(set, handler, net::LEARN_ALL);
     std::cout << "validation set\n";
     net::set validation = shapes.generate_shapes(100, 50, net::LEARN_ALL);
     shapes.check_shapes(validation, handler, net::LEARN_ALL);

     return 0;
}
#endif