import netStandalone
import os
import numpy as np
from schmidt import BaseElement
from schmidt import plotter as plt

PATH = os.path.join(os.environ['USERPROFILE'], "Desktop/CE")
handler = netStandalone.handler(PATH)

# Paso de integración para los productos escalares
step = 0.1
# Dominio de las funciones [-domaine,domaine]
domaine = 128
# Orden, para distribuir los bias se va dividiendo el dominio de dos en dos -> 2^order neuronas base
order = 5

yfunc = np.zeros(int(2*domaine/step))
yfunc = np.sin(np.arange(-domaine,domaine,step)/10)+1

myElement = BaseElement.myfunc(yfunc)
myBase = BaseElement.base(step, domaine, order)

myBase.graph_base()
# coefs = myBase.proyect_function(myElement)
# myBase.graph_representation(yfunc, coefs)

entradas = 5
npl = [myBase.max_index,myBase.max_index,1]

myBase.write_params(PATH+"/pyparams.csv",entradas,npl,[myElement])

ins = entradas
npl = [myBase.max_index,1]
act = [netStandalone.RELU2, netStandalone.RELU2]

handler.instantiate("cpu_float_test", netStandalone.CPU)
handler.set_active_net("cpu_float_test")
handler.build_net_from_file("pyparams",netStandalone.RELOAD_FILE)

X = np.arange(-domaine,domaine,step)
Y = []
for i in range(myBase.max_index):
    Y.append([])


for x in X:
    test_input = netStandalone.v_float([x,x,x,x,x])
    out = handler.run_forward(test_input)
    for i in range(myBase.max_index):
        Y[i].append(out[i])

args = {'title': "SCHMIDT",
            'x_label': "x", 'y_label': "y"}
plt.multiple_plot_and_wait(X, Y,args)
#handler.build_net_from_data(ins,netStandalone.v_int(npl),netStandalone.v_int(act))



#handler.build_net_from_file(,True)
handler.write_net_to_file("with_params")

#handler.build_net_from_data(ins,netStandalone.v_int(npl),netStandalone.v_int(act))
#tic = time.perf_counter()
#res = handler.run_forward(test_input)
#bench_list.append(time.perf_counter()-tic)
#handler.write_net_to_file(net_name+"_with_params")