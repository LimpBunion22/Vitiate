import netStandalone
import tensorflow as tf
import os

PATH = os.path.join(os.environ["HOME"], "workspace_development")
handler = netStandalone.handler(PATH)
handler.instantiate("net0", netStandalone.GPU)
handler.set_active_net("net0")

ins = netStandalone.v_float([1, 1])
handler.set_input_size(5)
handler.build_fully_layer(1000)
handler.build_fully_layer(1000)
handler.build_fully_layer(1000)
handler.build_fully_layer(1000)
handler.build_fully_layer(1000)
handler.build_net()

# handler.run_forward(netStandalone.v_float([1, 2, 3, 4, 5]))
# print(handler.get_forward_performance())
# handler.run_forward(netStandalone.v_float([1, 2, 3, 4, 5]))
# print(handler.get_forward_performance())

handler.build_net_from_file("net", netStandalone.REUSE_FILE)
handler.attr(netStandalone.EPOCHS, 20)\
    .attr(netStandalone.BATCH_SIZE, 64)\
    .attr(netStandalone.ALPHA, 30.0)\
    .attr(netStandalone.ALPHA_DECAY, 0.00001)\
    .attr(netStandalone.ERROR_THRESHOLD, 0.00001)\
    .attr(netStandalone.ABS)

print(handler.run_gradient("set", netStandalone.REUSE_FILE))
print(handler.get_gradient_performance())
