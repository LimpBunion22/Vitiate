import vitiate
import tensorflow as tf

handler = vitiate.net_handler("/home/hai/workspace_development")
handler.net_create("cpu_float", vitiate.CPU,
                   vitiate.DERIVATE, vitiate.NOT_RANDOM, "net")
handler.set_active_net("cpu_float")
# handler.active_net_init_gradient("sets")
# print(handler.active_net_launch_gradient(45))
# print(handler.active_net_get_gradient_performance())

ins = vitiate.v_data_type([1, 2])
print(handler.active_net_launch_forward(ins))
print(handler.active_net_get_forward_performance())

# handler.write_net_to_file("new_net")
