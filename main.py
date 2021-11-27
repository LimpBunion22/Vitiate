import vitiate

vitiate.rand_init(repeteable=True)
structure = vitiate.net_structure([3, 4, 5])
input1 = vitiate.set_single([2, 1])
set_ins = vitiate.set([input1])
out1 = vitiate.set_single([0, 0, 0, 0, 0])
set_outs = vitiate.set([out1])
net = vitiate.net_float(n_ins=2, n_p_l=structure, derivate=True)
net.init_gradient(set_ins, set_outs)
net.launch_gradient(iterations=10)
net.print_inner_vals()
print("gradient took", net.get_gradient_performance(), "us")
test_input1 = input1
net.launch_forward(test_input1)
print("forward took", net.get_forward_performance(), "us")
net.print_inner_vals()

