
import numpy as np
from schmidt.plotter import multiple_plot_and_wait
import os
import random

class base:

    def __init__(self, step, domaine, order):

        self.max_index = (order-1)*2**(order-1) + 3
        self.step = step
        self.domaine = domaine
        self.original_list = []
        self.base_list = []
        
        self.original_list.append(element(step, domaine, 0))
        self.original_list.append(element(step, domaine, domaine))
        self.original_list.append(element(step, domaine, -domaine))
        cnt = 3

        for o in range(1,order):
            for i in range(0,2**(o-1)):
                bias = domaine*(1 + 2*i)/(2**o)
                self.original_list.append(element(step, domaine, bias))
                self.original_list.append(element(step, domaine, -bias))
                cnt += 1

        self.max_index = cnt
        for i in range(0,self.max_index):
            self.base_list.append(base_element(i, self.max_index, self.base_list, self.original_list, step, domaine))

    def proyect_function(self, func):

        coefs = np.zeros(self.max_index)

        for i in range(0,self.max_index):
            coefs[i] = self.base_list[i].eval_proyection(func)

        return coefs

    def graph_base(self):

        xgraph = np.arange(-self.domaine, self.domaine, self.step)
        ygraph = []
        
        for i in range(self.max_index):
            ygraph.append(self.base_list[i].values)

        multiple_plot_and_wait(xgraph, ygraph, {})

    def graph_base_element(self, index):

        xgraph = np.arange(-self.domaine, self.domaine, self.step)
        ygraph = []
        ygraph.append(self.base_list[index].values)

        multiple_plot_and_wait(xgraph, ygraph, {})

    def graph_representation(self, yfunc, coefs):

        xgraph = np.arange(-self.domaine, self.domaine, self.step)
        ygraph = np.zeros(int(2*self.domaine/self.step))

        for x in np.arange(-self.domaine,self.domaine,self.step):
            ygraph[int((x + self.domaine)/self.step)] = 0
            for i in range(0,self.max_index):
                ygraph[int((x + self.domaine)/self.step)] += coefs[i]*self.base_list[i].values[int((x + self.domaine)/self.step)]

        multiple_plot_and_wait(xgraph, [yfunc, ygraph], {})

    def write_params(self, path_file, entradas, npl, funcs):
        if len(npl)<2:
            print("ERROR: layers number has to be 3 at least")
            return
        
        with open(path_file, "w") as file:
            file.write("//input size, neurons per layer\n"+str(entradas)+",")
            for n in range(len(npl)):
                file.write(str(npl[n])+",")
            # for n in npl:
            #     file.write(str(n)+",")
            file.write("\n\n//activations\n")
            for n in range(len(npl)):
                file.write("R2,")
            file.write("\n\n//params + bias (interleaved)\n")
            
            for n in range(npl[0]):
                file.write("1,")
                for e in range(entradas-1):
                    file.write("0,")  
            for n in range(npl[0]):              
                file.write(str(self.original_list[n].bias)+",")

            # for n in range(npl[1]):
            #     for e in range(npl[0]):
            #         file.write(str(self.base_list[n].coefs_base[e])+",")
            # for n in range(npl[1]):
            #     file.write("0,")                
                    
            for n in range(npl[1]):
                if n>=len(funcs):
                    for e in range(npl[0]):
                        file.write(str(random.random()-0.5)+",")
                else:
                    coefs_b = self.proyect_function(funcs[n])
                    coefs = np.zeros(len(coefs_b))
                    for c in range(len(coefs)):
                        for cn in range(len(coefs)):
                            coefs[c] += self.base_list[cn].coefs_base[c]*coefs_b[cn]

                    for e in range(npl[0]):
                        file.write(str(coefs[e])+",")

            for n in range(npl[1]):
                if n>len(funcs):
                    file.write(str(random.random()-0.5)+",")
                else:
                    file.write("0,")

            if len(npl)>2:
                for l in range(3,len(npl)):
                    for n in range(npl[l]):
                        for e in range(npl[l-1]+1):
                            file.write(str(random.random()-0.5)+",")
class base_element:

    def __init__(self, index, max_index, base_list, original_list, step, domaine):

        self.index = index
        self.max_index = max_index
        self.values = np.zeros(int(domaine*2/step))
        self.proyections = np.zeros(max_index)
        self.coefs_base = np.zeros(self.max_index)

        self.base_list = base_list
        self.original_list = original_list

        self.step = step
        self.domaine = domaine

        #eval base element values
        self.eval_base_value()

        #eval proyections over the element
        for i in range(index+1,max_index):
            self.proyections[i] = self.eval_proyection(original_list[i])

    def eval_base_value(self):

        for x in np.arange(-self.domaine, self.domaine, self.step):

            ind = int((x + self.domaine)/self.step)
            self.values[ind] = self.original_list[self.index].values[ind]

            for i in np.arange(self.index-1,-1,-1):
                self.values[ind] -= self.base_list[i].proyections[self.index]*self.base_list[i].values[ind]

        self.norma = 1/np.sqrt(np.sum(self.values*self.values)*self.step)
        self.values = self.values*self.norma

        self.coefs_base[self.index] = self.norma
        for i in np.arange(self.index-1,-1,-1):
            for c in np.arange(i,self.index):
                self.coefs_base[i] -= self.base_list[c].proyections[self.index]*self.base_list[c].coefs_base[i]
            self.coefs_base[i] = self.norma*self.coefs_base[i]

    def eval_proyection(self, in_func):

        aux = 0

        for x in np.arange(-self.domaine, self.domaine, self.step):
            ind = int((x + self.domaine)/self.step)
            aux += in_func.values[ind]*self.values[ind]

        return aux*self.step


class element:

    def __init__(self, step, domaine, bias):

        self.values = np.zeros(int(domaine*2/step))
        self.bias = bias

        for x in np.arange(-domaine,domaine,step):
            aux = x + bias
            if aux > 0:
                self.values[int((x + domaine)/step)] = aux
            else:
                self.values[int((x + domaine)/step)] =aux*0.0625

class myfunc:

    def __init__(self, yfunc):

        self.values = yfunc