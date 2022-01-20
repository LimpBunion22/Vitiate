import netStandalone
import tensorflow as tf

handler = netStandalone.net_handler("/home/gabi/workspace_development")
ins = netStandalone.v_data_type([1, 2])

# handler.net_create("cpu_float", netStandalone.CPU,
#                    netStandalone.DERIVATE, netStandalone.NOT_RANDOM, "net", file_reload=True)
handler.net_create_random_from_vector(
    "cpu_float", netStandalone.CPU, 2, n_p_l=netStandalone.v_size_t([3, 4, 5]))
handler.set_active_net("cpu_float")
handler.active_net_init_gradient("sets")
print(handler.active_net_launch_gradient(
    50, error_threshold=0.0001, multiplier=1.01))
print(handler.active_net_get_gradient_performance())
print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())
image_data = netStandalone.image_set()
handler.filter_image(image_data)
handler.get_filtered_image()

# handler.net_create("fpga_float", netStandalone.FPGA,
#                    netStandalone.DERIVATE, netStandalone.NOT_RANDOM, "net", file_reload=True)
# handler.set_active_net("fpga_float")
# print(handler.active_net_launch_forward(ins))
# print(handler.active_net_get_forward_performance())
