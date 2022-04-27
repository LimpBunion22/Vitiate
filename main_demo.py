import netStandalone
import tensorflow as tf
import os

PATH = os.path.join(os.environ["HOME"], "workspace_development")
handler = netStandalone.handler(PATH)
ins = netStandalone.v_float([1, 1])

handler.instantiate("net0", netStandalone.CPU)
handler.set_active_net("net0")

handler.build_net_from_file("net", netStandalone.REUSE_FILE)
handler.attr(netStandalone.EPOCHS, 50)\
    .attr(netStandalone.BATCH_SIZE, 64)\
    .attr(netStandalone.ALPHA, 30.0)\
    .attr(netStandalone.ALPHA_DECAY, 0.00001)\
    .attr(netStandalone.ERROR_THRESHOLD, 0.00001)\
    .attr(netStandalone.ABS)

print(handler.run_gradient("set", netStandalone.REUSE_FILE))
print(handler.get_gradient_performance())