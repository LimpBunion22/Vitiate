from python import tests
# import logger


tests.test_backward()
# tests.test_forward()


# logger.log("test info", "INFO")
# logger.log("test warnings", "WARNING")
# logger.log("test error", "ERROR")
# logger.log("test fatal", "FATAL")
# logger.log("test others")


# --------------------------------------------------------------------------
# import os
# import numpy as np
# import time
# from tqdm import tqdm
# import copy

# import netStandalone
# from utils import decorators
# from utils import plotter as plt
# from utils import logger
# import keras

# ITERACIONES = 500

# PATH = os.path.join(os.environ['HOME'], "workspace_development")
# PATH_NET = os.path.join( PATH,"_temporal_net.csv")

# inputs = 1
# with open(PATH_NET, "w") as file:
#     file.write(f"{inputs} 64 64 64 64 64 64 64 1 ")

# handler = netStandalone.net_handler(PATH)
# test_input = netStandalone.v_float(np.random.rand(inputs))

# handler.net_create("cpu_float_test", netStandalone.CPU, netStandalone.NOT_DERIVATE, netStandalone.RANDOM, "_temporal_net", file_reload = True)
# handler.set_active_net("cpu_float_test")
# tic = time.perf_counter()
# for i in range(ITERACIONES):
#     handler.active_net_launch_forward(test_input)
# cpu_time = time.perf_counter()-tic
# logger.log("CPU " + str(cpu_time/ITERACIONES), "INFO")
# handler.active_net_write_net_to_file("_temporal_net_with_params")

# handler.net_create("fpga_float_test", netStandalone.FPGA, netStandalone.NOT_DERIVATE, netStandalone.NOT_RANDOM, "_temporal_net_with_params", file_reload = True)
# handler.set_active_net("fpga_float_test")
# handler.active_net_launch_forward(test_input)
# tic = time.perf_counter()
# for i in range(ITERACIONES):
#     handler.active_net_launch_forward(test_input)
# fpga_time = time.perf_counter()-tic
# logger.log("FPGA " + str(fpga_time/ITERACIONES), "INFO")
# logger.log("FPGA/CPU " + str(fpga_time/cpu_time), "INFO")

# --------------------------------------------------------------------------

# import tensorflow as tf
# import netStandalone
# import cv2
# import numpy as np
# import os
# import time
# from utils import logger

# cap = cv2.VideoCapture("/home/hai/Downloads/VideoDAE.mp4")
# ret, frame = cap.read()
# # while(1):
# #    ret, frame = cap.read()
# #    cv2.imshow('frame',frame)
# #    if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
# #        cap.release()
# #        cv2.destroyAllWindows()
# #        break
# #    cv2.imshow('frame',frame)

# for i in range(10):
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
#         cap.release()
#         cv2.destroyAllWindows()
#     cv2.imshow('frame', frame)

# PATH = os.path.join(os.environ['HOME'], "workspace_development")
# PATH_NET = os.path.join( PATH,"_temporal_net.csv")
# inputs = 1
# with open(PATH_NET, "w") as file:
#     file.write(f"{inputs} 64 64 64 64 64 64 64 1 ")

# handler = netStandalone.net_handler(PATH)
# handler.net_create("fpga_float_test", netStandalone.FPGA, netStandalone.NOT_DERIVATE, netStandalone.NOT_RANDOM, "_temporal_net_with_params", file_reload = True)
# handler.set_active_net("fpga_float_test")


# image_data = netStandalone.image_set()
# data = netStandalone.v_uchar(1080*1920*[0])

# image_data.original_x_pos = 0
# image_data.original_y_pos = 0
# image_data.original_h = 1080
# image_data.original_w = 1920

# for i in range(1080):
#     for j in range(1920):
#         data[i*1920+j] = int(frame[i][j][0]/3 + frame[i][j][1]/3 + frame[i][j][2]/3)

# image_data.resized_image_data = data

# # tic = time.perf_counter()
# # for i in range(ITERACIONES):
# #     handler.active_net_launch_forward(test_input)
# # cpu_time = time.perf_counter()-tic
# # logger.log("CPU " + str(cpu_time/ITERACIONES), "INFO")
# handler.filter_image(image_data)
# image_data = handler.get_filtered_image()

# # tic = time.perf_counter()
# # for i in range(20):
# #     for b in range(24):
# #         handler.filter_image(image_data)
# #     for b in range(24):
# #         image_data = handler.get_filtered_image()
# # exe_time = time.perf_counter()-tic
# # logger.log("Imagenes por segundo: " + str((24*20)/exe_time), "INFO")

# # handler.filter_image(image_data)
# # image_data = handler.get_filtered_image()

# for i in range(1080):
#     for j in range(1920):
#         frame[i][j][0] = image_data.resized_image_data[i*1920+j]*6
#         frame[i][j][1] = image_data.resized_image_data[i*1920+j]*6
#         frame[i][j][2] = image_data.resized_image_data[i*1920+j]*6

# cv2.imshow('frame', frame)
# if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
#     cap.release()
#     cv2.destroyAllWindows()

# print("Finish")

# --------------------------------------------------------------------------

import netStandalone
import os


# handler = netStandalone.net_handler("/home/hai/workspace_development")
# ins = netStandalone.v_float([1, 2])

# # handler.net_create("cpu_float", netStandalone.CPU,
# #                    netStandalone.DERIVATE, netStandalone.NOT_RANDOM, "net", file_reload=True)
# handler.net_create_random_from_vector(
#     "cpu_float", netStandalone.CPU, 2, n_p_l=netStandalone.v_size_t([100, 1000, 1000, 5]))
# handler.set_active_net("cpu_float")
# handler.active_net_init_gradient("sets")
# print(handler.active_net_launch_gradient(
#     50, error_threshold=0.0001, multiplier=1.01))
# print(handler.active_net_get_gradient_performance())
# print(handler.active_net_launch_forward(ins))
# print(handler.active_net_get_forward_performance())

# handler.net_create_random_from_vector(
#     "gpu_float", netStandalone.GPU, 2, n_p_l=netStandalone.v_size_t([100, 1000, 1000, 5]))
# handler.set_active_net("gpu_float")
# handler.active_net_init_gradient("sets")
# handler.active_net_launch_forward(ins)
# print(handler.active_net_launch_gradient(
#     50, error_threshold=0.0001, multiplier=1.01))
# print(handler.active_net_get_gradient_performance())
# print(handler.active_net_launch_forward(ins))
# print(handler.active_net_get_forward_performance())



# PATH = os.path.join(os.environ['HOME'], "workspace_development")
# PATH_NET = os.path.join( PATH,"_temporal_net.csv")
# inputs = 1
# with open(PATH_NET, "w") as file:
#     file.write(f"{inputs}1,1,")

# handler = netStandalone.net_handler(PATH)
# handler.net_create("fpga_float_test", netStandalone.FPGA, netStandalone.FIXED, "_temporal_net_with_params", file_reload = True)
# handler.set_active_net("fpga_float_test")

# handler.process_video("VideoDAE.mp4")
# print("Finish")
