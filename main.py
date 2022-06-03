from python import tests
from python.utils import logger


# tests.test_backward()
tests.test_forward()


# #logger.log("test info", "INFO")
# #logger.log("test warnings", "WARNING")
# #logger.log("test error", "ERROR")
# #logger.log("test fatal", "FATAL")
# #logger.log("test others")


# --------------------------------------------------------------------------
# import os
# # import numpy as np
# import time
# # from tqdm import tqdm
# # import copy

# import netStandalone
# # from utils import decorators
# # from utils import plotter as plt
# # from utils import logger
# # import keras

# PATH = os.path.join(os.environ["HOME"], "workspace_development")
# ins = netStandalone.v_float([1, 3])

# #logger.log("PYTHON: Creating handler", "INFO")
# handler = netStandalone.handler(PATH)
# #logger.log("PYTHON: Handler created", "INFO")

# # #logger.log("PYTHON: Creating net", "INFO")
# # handler.net_create(
# #     "cpu_float", netStandalone.CPU, netStandalone.FIXED_NET, "fpga_file_net", file_reload=netStandalone.REUSE_FILE)
# # handler.net_create(
# # "fpga_float", netStandalone.FPGA, netStandalone.FIXED_NET  , "fpga_file_net", file_reload=netStandalone.REUSE_FILE)
# handler.net_create_random_from_vector(
#     "fpga_float", netStandalone.FPGA, 2, netStandalone.v_size_t([1000, 1000, 1000, 1000, 1000]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2]))
# # #logger.log("PYTHON: Net created", "INFO")

# # #logger.log("PYTHON: Activating net", "INFO")
# # handler.set_active_net("fpga_float")
# # #logger.log("PYTHON: Net activated", "INFO")

# # #logger.log("PYTHON: Launching forward", "INFO")
# # print("FPGA:\n")
# # tic = time.perf_counter()
# # handler.active_net_launch_forward(ins)
# # print(handler.active_net_launch_forward(ins))
# # logger.log("PYTHON: Total Forward Time"+str(time.perf_counter()-tic)+"ms\n", "INFO")
# # # handler.active_net_write_net_to_file("fpga_file_net")
# # #logger.log("PYTHON: Forward executed", "INFO")


# # handler.set_active_net("cpu_float")
# # print("CPU:")
# # print(handler.active_net_launch_forward(ins))

# print("\n\nTESTING CORES")
# POPULATION = 10
# handler.set_active_net("fpga_float")
# for i in range(POPULATION):
#     handler.net_create_random_from_vector("fpga_float_test_"+str(i), netStandalone.FPGA, 2, netStandalone.v_size_t([1000, 1000, 1000, 1000, 1000]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2]))
#     # handler.net_create("fpga_float_test_"+str(i), netStandalone.FPGA, netStandalone.FIXED_NET  , "fpga_file_net", file_reload=netStandalone.REUSE_FILE)
#     # handler.enq_fpga_net("fpga_float_test_"+str(i),netStandalone.v_float([i, 3]),True,True)

# ITERATIONS = 40

# for it in range(ITERATIONS):
#     logger.log("------------------------------------------------------------------------------------------------------------------")
#     logger.log("ITERACION  "+str(it))
#     print("\ ENQ CORES\n")
#     for i in range(POPULATION):
#         # handler.net_create_random_from_vector("fpga_float_test_"+str(i), netStandalone.FPGA, 2, netStandalone.v_size_t([1000, 1000, 1000, 1000, 1000]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2]))
#         # handler.net_create("fpga_float_test_"+str(i), netStandalone.FPGA, netStandalone.FIXED_NET  , "fpga_file_net", file_reload=netStandalone.REUSE_FILE)
#         handler.enq_fpga_net("fpga_float_test_"+str(i),netStandalone.v_float([i, 3]),True,True)

#     print("\n EXE CORES\n")
#     tic = time.perf_counter()
#     handler.exe_fpga_nets()
#     logger.log("  PYTHON: Total EXE Time"+str(time.perf_counter()-tic)+"ms\n", "INFO")

#     print("\n READING CORES")
#     for i in range(POPULATION):
#         handler.read_fpga_net("fpga_float_test_"+str(i))


# --------------------------------------------------------------------------

# PATH = os.path.join(os.environ['HOME'], "workspace_development")
# PATH_NET = os.path.join( PATH,"_temporal_net.csv")
# inputs = 1
# with open(PATH_NET, "w") as file:
#     file.write(f"{inputs}1,1,")

# handler = netStandalone.handler(PATH)
# handler.net_create("fpga_float_test", netStandalone.FPGA, netStandalone.FIXED, "_temporal_net_with_params", file_reload = True)
# handler.set_active_net("fpga_float_test")

# handler.process_video("VideoDAE.mp4")
# print("Finish")
