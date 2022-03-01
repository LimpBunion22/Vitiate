import netStandalone
import tensorflow as tf
import os

PATH = os.path.join(os.environ["HOME"], "workspace_development")
handler = netStandalone.net_handler(PATH)
ins = netStandalone.v_float([1, 2])

handler.net_create_random_from_vector(
    "cpu2_float", netStandalone.CPU, 2, n_p_l=netStandalone.v_size_t([3, 4, 5]),
    activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2]))

handler.net_create(
    "cpu_float", netStandalone.CPU, netStandalone.FIXED, "net")
handler.set_active_net("cpu_float")
handler.active_net_init_gradient("sets")
print(handler.active_net_launch_gradient(
    50, error_threshold=0.0001, multiplier=1.01))
print(handler.active_net_get_gradient_performance())
print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())

handler.delete_net("cpu_float")
handler.active_net_launch_forward(ins)

# handler.net_create(
#     "gpu_float", netStandalone.GPU, netStandalone.FIXED, "net")
# handler.set_active_net("gpu_float")
# handler.active_net_init_gradient("sets")
# print(handler.active_net_launch_gradient(
#     50, error_threshold=0.0001, multiplier=1.01))
# print(handler.active_net_get_gradient_performance())
# print(handler.active_net_launch_forward(ins))
# print(handler.active_net_get_forward_performance())