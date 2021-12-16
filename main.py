import vitiate
import tensorflow as tf

handler = vitiate.net_handler("/home/gabi/workspace_development")
handler.net_create("cpu_float", vitiate.CPU,
                   vitiate.DERIVATE, vitiate.RANDOM, "net")
handler.init_gradient("cpu_float", "sets")
print(" gradient errors")
print(handler.launch_gradient("cpu_float", 45))
print(" gradient performance was ",
      handler.get_gradient_performance("cpu_float"), " us")

ins = vitiate.v_data_type([1, 2])
print(" forward outputs from ins ", ins)
print(handler.launch_forward("cpu_float", ins))
print(" forward performance was ",
      handler.get_forward_performance("cpu_float"), "us")

handler.write_net_to_file("cpu_float", "new_net")
