import netStandalone
import tensorflow as tf
import os

PATH = os.path.join(os.environ["HOME"], "workspace_development")
handler = netStandalone.net_handler(PATH)
ins = netStandalone.v_float([1, 1])

handler.net_create(
    "cpu_float", netStandalone.CPU, netStandalone.FIXED, "net", file_reload=netStandalone.REUSE_FILE)
# handler.net_create_random_from_vector(
#     "cpu_float", netStandalone.CPU, 2, netStandalone.v_size_t([100, 20, 5]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2_SOFT_MAX]))
handler.set_active_net("cpu_float")
print(handler.active_net_launch_gradient(
    iterations=15, batch_size=netStandalone.FULL_BATCH, alpha=2, alpha_decay=0.0001, reg_lambda=0.01,
    error_threshold=0, norm=netStandalone.NORM_REG_2, dropout_interval=0, file="sets", file_reload=netStandalone.REUSE_FILE))
print(handler.active_net_get_gradient_performance())
print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())

handler.net_create(
    "gpu_float", netStandalone.GPU, netStandalone.FIXED, "net", file_reload=netStandalone.REUSE_FILE)
# handler.net_create_random_from_vector(
#     "gpu_float", netStandalone.CPU, 2, netStandalone.v_size_t([100, 20, 5]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2_SOFT_MAX]))
handler.set_active_net("gpu_float")
print(handler.active_net_launch_gradient(
    iterations=15, batch_size=netStandalone.FULL_BATCH, alpha=2, alpha_decay=0.0001, reg_lambda=0.01,
    error_threshold=0, norm=netStandalone.NORM_REG_2, dropout_interval=0, file="sets", file_reload=netStandalone.REUSE_FILE))
print(handler.active_net_get_gradient_performance())
print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())
