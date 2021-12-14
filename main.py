import vitiate
import tensorflow as tf

vitiate.rand_init(repeatable=True)
file_manager = vitiate.net_float.file_manager()
file_manager.load_net_structure("net")
file_manager.load_sets("sets")
net = vitiate.net_float(file_manager, derivate=True, random=True)
net.init_gradient(file_manager)
print("gradient errors are", net.launch_gradient(iterations=65), "\n")
print("gradient took", net.get_gradient_performance(), "us\n")
test_input2 = vitiate.v_float([2, 1])
print("forward output is", net.launch_forward(test_input2), "\n")
print("forward took", net.get_forward_performance(), "us\n")
