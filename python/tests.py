import os
import numpy as np
import time
from tqdm import tqdm
import copy

import netStandalone
from python.utils import decorators
from python.utils import plotter as plt
from python.utils import logger
import keras

STEP = 160
TEST_DIM = {
    'inputs': 15*STEP,
    'layers': 15*STEP,
    'n_per_layer': 15*STEP
}

ITERATIONS = 5
IT_FORWARD = False

POPULATION = 10
POP_FORWARD = False

PATH = os.path.join(os.environ['HOME'], "workspace_development")
PATH_NET = os.path.join(PATH, "_temporal_net.csv")
PATH_SETS = os.path.join(PATH, "_temporal_data.csv")


# Evaluate fordward over diferent nets structures
def test_forward():

    structure = {
        'inputs_n': 5,
        'layers_n': 5,
        'neurons_per_layer': [5, 5, 5, 5, 5]
    }
    benchmarks = {
        'inputs': [],
        'layers': [],
        'n_per_layer': []
    }
    cpp_bench = benchmarks
    fpga_bench = copy.deepcopy(benchmarks)
    gpu_bench = copy.deepcopy(benchmarks)
    keras_bench = copy.deepcopy(benchmarks)
    handler = netStandalone.net_handler(PATH)

    # Inputs
    logger.log("Testing forward inputs", "INFO")
    for i in tqdm(range(5, TEST_DIM['inputs'], STEP)):
        structure['inputs_n'] = i
        with open(PATH_NET, "w") as file:
            file.write(f"{i},5,5,5,5,5,\n\nR,R,R,R,R,\n\n")

        test_input = netStandalone.v_float(np.random.rand(i))
        res_cpu = cpp_forward(cpp_bench['inputs'], handler, test_input)
        res_fpga = fpga_forward(fpga_bench['inputs'], handler, test_input)
        res_gpu = gpu_forward(gpu_bench['inputs'], handler, test_input)
        keras_forward(keras_bench['inputs'], structure)
        # validate_forward(res_cpu, res_fpga, "inputs ", i)

    print("\n")

    # Layers
    logger.log("Testing forward layers", "INFO")
    aux_string = "5,5,5,5,5,5,"
    aux_string_2 = "R,R,R,R,R,"
    structure['inputs_n'] = 5
    test_input = netStandalone.v_float(np.random.rand(5))
    for i in tqdm(range(5, TEST_DIM['layers'], STEP)):
        structure['layers_n'] = i
        structure['neurons_per_layer'] = 5*np.ones(i)
        with open(PATH_NET, "w") as file:
            file.write(aux_string+"\n\n"+aux_string_2+"\n\n")

        res_cpu = cpp_forward(cpp_bench['layers'], handler, test_input)
        res_fpga = fpga_forward(fpga_bench['layers'], handler, test_input)
        res_gpu = gpu_forward(gpu_bench['layers'], handler, test_input)
        keras_forward(keras_bench['layers'], structure)

        for j in range(STEP):
            aux_string += "5,"
            aux_string_2 += "R,"
        # validate_forward(res_cpu, res_fpga, "layers ", i)

    print("\n")

    # Neurons per layer
    logger.log("Testing forward neurons per layer", "INFO")
    structure['layers_n'] = 5
    aux_string_2 = "R,R,R,R,R,"
    test_input = netStandalone.v_float(np.random.rand(5))
    for i in tqdm(range(5, TEST_DIM['n_per_layer'], STEP)):
        ls = i*np.ones(5).astype(int)
        aux_string = "5,"+"".join(str(e)+"," for e in ls)
        structure['neurons_per_layer'] = i*np.ones(5)
        with open(PATH_NET, "w") as file:
            file.write(aux_string+"\n\n"+aux_string_2+"\n\n")

        res_cpu = cpp_forward(cpp_bench['n_per_layer'], handler, test_input)
        res_fpga = fpga_forward(fpga_bench['n_per_layer'], handler, test_input)
        res_gpu = gpu_forward(gpu_bench['n_per_layer'], handler, test_input)
        keras_forward(keras_bench['n_per_layer'], structure)
        # validate_forward(res_cpu, res_fpga, "npl ", i)
    print("\n")

    x_in = np.arange(5, TEST_DIM['inputs'], STEP)
    x_ly = np.arange(5, TEST_DIM['layers'], STEP)
    x_npl = np.arange(5, TEST_DIM['n_per_layer'], STEP)

    # y = [1000*np.array(cpp_bench['inputs']), 1000 *
    #      np.array(keras_bench['inputs']), 1000*np.array(gpu_bench['inputs'])]
    # args = {'title': "FORWARD: INPUTS BENCH",
    #         'x_label': "[Inputs number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "GPU"]}
    # plt.multiple_plot(x_in, y, args)

    # y = [1000*np.array(cpp_bench['layers']), 1000 *
    #      np.array(keras_bench['layers']), 1000*np.array(gpu_bench['layers'])]
    # args = {'title': "FORWARD: LAYERS BENCH",
    #         'x_label': "[Layers number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "GPU"]}
    # plt.multiple_plot(x_ly, y, args)

    # y = [1000*np.array(cpp_bench['n_per_layer']), 1000*np.array(
    #     keras_bench['n_per_layer']), 1000*np.array(gpu_bench['n_per_layer'])]
    # args = {'title': "FORWARD: NEURONS PER LAYER BENCH",
    #         'x_label': "[Neurons per layer number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "GPU"]}
    # plt.multiple_plot_and_wait(x_npl, y, args)

    y = [1000*np.array(cpp_bench['inputs']), 1000*np.array(keras_bench['inputs']),
         1000*np.array(fpga_bench['inputs']), 1000*np.array(gpu_bench['inputs'])]
    args = {'title': "FORWARD: INPUTS BENCH",
            'x_label': "[Inputs number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "FPGA", "GPU"]}
    plt.multiple_plot(x_in, y, args)

    y = [1000*np.array(cpp_bench['layers']), 1000*np.array(keras_bench['layers']),
         1000*np.array(fpga_bench['layers']), 1000*np.array(gpu_bench['layers'])]
    args = {'title': "FORWARD: LAYERS BENCH",
            'x_label': "[Layers number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "FPGA", "GPU"]}
    plt.multiple_plot(x_ly, y, args)

    y = [1000*np.array(cpp_bench['n_per_layer']), 1000*np.array(keras_bench['n_per_layer']),
         1000*np.array(fpga_bench['n_per_layer']), 1000*np.array(gpu_bench['n_per_layer'])]
    args = {'title': "FORWARD: NEURONS PER LAYER BENCH",
            'x_label': "[Neurons per layer number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "FPGA", "GPU"]}
    plt.multiple_plot_and_wait(x_npl, y, args)

    return

# Evaluate backward over diferent nets structures


def test_backward():

    structure = {
        'inputs_n': 5,
        'layers_n': 5,
        'neurons_per_layer': [5, 5, 5, 5, 5]
    }
    benchmarks = {
        'inputs': [],
        'layers': [],
        'n_per_layer': []
    }
    cpp_bench = benchmarks
    gpu_bench = copy.deepcopy(benchmarks)
    keras_bench = copy.deepcopy(benchmarks)
    handler = netStandalone.net_handler(PATH)

    # Inputs
    aux_string_d = "1,\n\n1,2,3,4,5,"
    logger.log("Testing backward inputs", "INFO")
    for i in tqdm(range(5, TEST_DIM['inputs'], STEP)):
        structure['inputs_n'] = i
        with open(PATH_NET, "w") as file:
            file.write(f"{i},5,5,5,5,5,\n\nR,R,R,R,R,\n\n")

        with open(PATH_SETS, "w") as file:
            file.write(aux_string_d + "\n" + "5,5,5,5,5,")

        for j in range(STEP):
            aux_string_d += str(1+i+j) + ","

        cpp_backward(cpp_bench['inputs'], handler)
        gpu_backward(gpu_bench['inputs'], handler)
        keras_backward(keras_bench['inputs'], structure)

    print("\n")

    # Layers
    aux_string = "5,5,5,5,5,"
    aux_string_2 = "R,R,R,R,R,"
    structure['inputs_n'] = 5
    logger.log("Testing backward layers", "INFO")
    with open(PATH_SETS, "w") as file:
        file.write("1,\n\n1,2,3,4,5," + "\n" + "5,5,5,5,5,")

    for i in tqdm(range(5, TEST_DIM['layers'], STEP)):
        structure['layers_n'] = i
        structure['neurons_per_layer'] = 5*np.ones(i)
        with open(PATH_NET, "w") as file:
            file.write(aux_string+"\n\n"+aux_string_2+"\n\n")

        cpp_backward(cpp_bench['layers'], handler)
        gpu_backward(gpu_bench['layers'], handler)
        keras_backward(keras_bench['layers'], structure)

        for j in range(STEP):
            aux_string += "5,"
            aux_string_2 += "R,"

    print("\n")

    # Neurons per layer
    aux_string_d = "5,5,5,5,5,"
    aux_string_2 = "R,R,R,R,R,"
    logger.log("Testing backward neurons per layer", "INFO")
    structure['layers_n'] = 5
    for i in tqdm(range(5, TEST_DIM['n_per_layer'], STEP)):
        ls = i*np.ones(5).astype(int)
        aux_string = "5,"+"".join(str(e)+"," for e in ls)
        structure['neurons_per_layer'] = i*np.ones(5)
        with open(PATH_NET, "w") as file:
            file.write(aux_string+"\n\n"+aux_string_2+"\n\n")

        with open(PATH_SETS, "w") as file:
            file.write("1,\n\n1,2,3,4,5," + "\n" + aux_string_d)

        for j in range(STEP):
            aux_string_d += str(1+i+j) + ","

        cpp_backward(cpp_bench['n_per_layer'], handler)
        gpu_backward(gpu_bench['n_per_layer'], handler)
        keras_backward(keras_bench['n_per_layer'], structure)

    print("\n")

    x_in = np.arange(5, TEST_DIM['inputs'], STEP)
    x_ly = np.arange(5, TEST_DIM['layers'], STEP)
    x_npl = np.arange(5, TEST_DIM['n_per_layer'], STEP)

    y = [1000*np.array(cpp_bench['inputs']), 1000 *
         np.array(keras_bench['inputs']), 1000*np.array(gpu_bench['inputs'])]
    args = {'title': "BACKWARD: INPUTS BENCH",
            'x_label': "[Inputs number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "GPU"]}
    plt.multiple_plot(x_in, y, args)

    y = [1000*np.array(cpp_bench['layers']), 1000 *
         np.array(keras_bench['layers']), 1000*np.array(gpu_bench['layers'])]
    args = {'title': "BACKWARD: LAYERS BENCH",
            'x_label': "[Inputs number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "GPU"]}
    plt.multiple_plot(x_ly, y, args)

    y = [1000*np.array(cpp_bench['n_per_layer']), 1000*np.array(
        keras_bench['n_per_layer']), 1000*np.array(gpu_bench['n_per_layer'])]
    args = {'title': "BACKWARD: NEURONS PER LAYER BENCH",
            'x_label': "[Inputs number]", 'y_label': "[ms]", 'label': ["Cpp", "Keras", "GPU"]}
    plt.multiple_plot_and_wait(x_npl, y, args)

    return


def cpp_forward(bench_list, handler, test_input, net_name="_temporal_net"):

    handler.net_create("cpu_float_test", netStandalone.CPU,
                       netStandalone.RANDOM, net_name, file_reload=True)
    handler.set_active_net("cpu_float_test")
    tic = time.perf_counter()
    res = handler.active_net_launch_forward(test_input)
    bench_list.append(time.perf_counter()-tic)
    handler.active_net_write_net_to_file(net_name+"_with_params")
    return res


def fpga_forward(bench_list, handler, test_input, net_name="_temporal_net"):

    handler.net_create("fpga_float_test", netStandalone.FPGA,
                    netStandalone.FIXED, net_name+"_with_params", file_reload=True)
    if(POP_FORWARD):
        for i in range(POPULATION):
            handler.net_create("fpga_float_test_"+str(i), netStandalone.FPGA,
                            netStandalone.FIXED, net_name+"_with_params", file_reload=True)

    handler.set_active_net("fpga_float_test")
    
    if(POP_FORWARD):
        tic = time.perf_counter()
        handler.enq_fpga_net("fpga_float_test",test_input)
        for i in range(POPULATION):
            handler.enq_fpga_net("fpga_float_test_"+str(i),test_input,True,True)

        handler.exe_fpga_nets()

        res = handler.read_fpga_net("fpga_float_test")
        for i in range(POPULATION):
            handler.read_fpga_net("fpga_float_test_"+str(i))

        if(IT_FORWARD):
            for it in range(ITERATIONS):
                handler.enq_fpga_net("fpga_float_test",test_input,False)
                for i in range(POPULATION):
                    handler.enq_fpga_net("fpga_float_test_"+str(i),test_input,False,True)

                handler.exe_fpga_nets()

                res = handler.read_fpga_net("fpga_float_test")
                for i in range(POPULATION):
                    handler.read_fpga_net("fpga_float_test_"+str(i))
            bench_list.append((time.perf_counter()-tic)/((POPULATION+1)*(ITERATIONS+1)))
        else:
            bench_list.append((time.perf_counter()-tic)/(POPULATION+1))
    else:
        tic = time.perf_counter()
        res = handler.active_net_launch_forward(test_input)
        if(IT_FORWARD):
            for it in range(ITERATIONS):
                handler.active_net_launch_forward(test_input)
        bench_list.append((time.perf_counter()-tic)/(ITERATIONS+1))
    return res


def gpu_forward(bench_list, handler, test_input, net_name="_temporal_net"):

    handler.net_create("gpu_float_test", netStandalone.GPU,
                       netStandalone.FIXED, net_name+"_with_params", file_reload=True)
    handler.set_active_net("gpu_float_test")
    tic = time.perf_counter()
    res = handler.active_net_launch_forward(test_input)
    bench_list.append(time.perf_counter()-tic)
    return res


def keras_forward(bench_list, structure):

    keras_model = keras.create_keras_net(
        structure['inputs_n'], structure['layers_n'], structure['neurons_per_layer'])
    test_input = keras.create_test_input(structure['inputs_n'])
    tic = time.perf_counter()
    keras_model(test_input)
    bench_list.append(time.perf_counter()-tic)
    return


def cpp_backward(bench_list, handler, net_name="_temporal_net", set_name="_temporal_data"):

    handler.net_create("cpu_float_test", netStandalone.CPU,
                       netStandalone.RANDOM, net_name, file_reload=True)
    handler.set_active_net("cpu_float_test")
    # handler.active_net_init_gradient(set_name, file_reload=True)
    tic = time.perf_counter()
    handler.active_net_launch_gradient(iterations=ITERATIONS, batch_size=netStandalone.FULL_BATCH, alpha=2, alpha_decay=0.001, reg_lambda=0.01,
                                       error_threshold=0.0001, norm=netStandalone.NORM_REG_2, dropout_interval=0, file=set_name, file_reload=netStandalone.RELOAD_FILE)
    bench_list.append((time.perf_counter()-tic)/ITERATIONS)
    return


def gpu_backward(bench_list, handler, net_name="_temporal_net", set_name="_temporal_data"):

    handler.net_create("gpu_float_test", netStandalone.GPU,
                       netStandalone.RANDOM, net_name, file_reload=True)
    handler.set_active_net("gpu_float_test")
    # handler.active_net_init_gradient(set_name, file_reload=True)
    tic = time.perf_counter()
    handler.active_net_launch_gradient(iterations=ITERATIONS, batch_size=netStandalone.FULL_BATCH, alpha=2, alpha_decay=0.001, reg_lambda=0.01,
                                       error_threshold=0.0001, norm=netStandalone.NORM_REG_2, dropout_interval=0, file=set_name, file_reload=netStandalone.RELOAD_FILE)
    bench_list.append((time.perf_counter()-tic)/ITERATIONS)
    return


def keras_backward(bench_list, structure):

    keras_model = keras.create_keras_net(
        structure['inputs_n'], structure['layers_n'], structure['neurons_per_layer'])
    tic = time.perf_counter()
    keras.run_backward(keras_model, structure, ITERATIONS)
    bench_list.append(time.perf_counter()-tic)
    return


def validate_forward(res_cpu, res_fpga, error_msg, error_index):

    rnd_cpu = np.round(res_cpu, 2)
    rnd_fpga = np.round(res_fpga, 2)

    if np.sum(rnd_cpu == rnd_fpga) != len(res_fpga):
        error_distance = np.sum(np.abs(np.abs(rnd_cpu)-np.abs(rnd_fpga)))
        if error_distance < 1:
            logger.log("FPGA FORWARD FAIL: " + error_msg + str(error_index) +
                       "  ERROR_DISTANCE: " + str(error_distance), "WARNING")
        else:
            logger.log("FPGA FORWARD FAIL: " + error_msg + str(error_index) +
                       "  ERROR_DISTANCE: " + str(error_distance), "ERROR")

    return
