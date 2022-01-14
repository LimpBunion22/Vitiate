import netStandalone
import tensorflow as tf

handler = netStandalone.net_handler("/home/hai/workspace_development")
ins = netStandalone.v_data_type([0, 0])

handler.net_create("cpu_float", netStandalone.CPU,
                   netStandalone.DERIVATE, netStandalone.NOT_RANDOM, "net", file_reload=True)
handler.set_active_net("cpu_float")
print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())

handler.net_create("fpga_float", netStandalone.FPGA,
                   netStandalone.DERIVATE, netStandalone.NOT_RANDOM, "net", file_reload=True)
handler.set_active_net("fpga_float")
# handler.active_net_init_gradient("sets")
# print(handler.active_net_launch_gradient(45))
# print(handler.active_net_get_gradient_performance())

print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())
