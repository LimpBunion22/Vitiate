from struct import pack
import netStandalone
import numpy as np
import copy
import os
from tqdm import tqdm
from logger import log

PATH = os.path.join(os.environ['HOME'], "workspace_development")
SETS_NAME = "_temporal_sets"
ADDIT_SETS = ["_cuadrados", "_triangulos", "_circulos"]
PATH_SETS = os.path.join(PATH, SETS_NAME + ".csv")
ADDIT_PATHS = []
for i in range(len(ADDIT_SETS)):
    ADDIT_PATHS.append(os.path.join(PATH, SETS_NAME + ADDIT_SETS[i] + ".csv"))

GRADIENT_ITERATIONS = 4

class ag_handler:

    def __init__(self, population_size, n_ins, n_outs, net_imp = netStandalone.CPU):

        self.gen = 0
        self.pop_size = population_size
        self.names = []
        self.archs = []
        self.scores = np.zeros(self.pop_size)
        self.black_list = np.zeros(self.pop_size)
        self.imp = net_imp
        self.n_ins = n_ins

        self.handler = netStandalone.net_handler(PATH)
        for p in range(self.pop_size):
            # neurons = [np.random.randint(n_ins,n_ins +50), np.random.randint(int(n_ins/2),int(n_ins/2) + 50), np.random.randint(n_outs,n_outs + 50), n_outs]
            self.archs.append([np.random.randint(200,1000), np.random.randint(10,100), n_outs])
            activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2_SOFT_MAX]) #RELU2_SOFT_MAX
            self.names.append("AG_NET_G0_" + str(p))
            self.handler.net_create_random_from_vector(self.names[p], net_imp, n_ins, n_p_l=netStandalone.v_size_t(self.archs[p]),activation_type=activation_type)
        
        self._data_out = []    
        self._pack_data_out = []

        return


    def exe_all(self, data_in, data_out = []):

        while(len(data_out) < self.pop_size):
            data_out.append([])

        for p in range(self.pop_size):
            self.handler.set_active_net(self.names[p])
            data_out[p] = self.handler.active_net_launch_forward(netStandalone.v_float(data_in))

        return data_out


    def exe_pack_all(self, pack_data_in, pack_data_out = []):

        if len(pack_data_out) != self.pop_size:
            while(len(pack_data_out) < self.pop_size):
                pack_data_out.append([])

            while(len(pack_data_out[0]) < len(pack_data_in)):
                pack_data_out[0].append([])

            for p in range(1,self.pop_size):
                pack_data_out[p] = copy.deepcopy(pack_data_out[0])

        for p in range(self.pop_size):
            self.handler.set_active_net(self.names[p])

            for i in range(len(pack_data_in)):
                pack_data_out[p][i] = self.handler.active_net_launch_forward(netStandalone.v_float(pack_data_in[i]))

        return pack_data_out


    def exe_best(self, data_in):

        self.handler.set_active_net(self.names[self.black_list[0]])

        return self.handler.active_net_launch_forward(netStandalone.v_float(data_in))

    def exe_pack_best(self, pack_data_in):

        pack_data_out = []
        self.handler.set_active_net(self.names[self.black_list[0]])

        for i in range(len(pack_data_in)):
            pack_data_out.append(self.handler.active_net_launch_forward(netStandalone.v_float(pack_data_in[i])))

        return pack_data_out


    def write_training_file(self, pack_data_in, pack_rigth_outs):        

        addit_str = ["","",""]
        addit_cnt = [0,0,0]
        print("Writing file")
        with open(PATH_SETS, "w") as file:
            file.write(f"{len(pack_data_in)} \n\n")

            for i in range(len(pack_data_in)):
                aux_str_in = ""
                aux_str_out = ""

                for j in range(len(pack_data_in[0])):
                    aux_str_in += str(pack_data_in[i][j]) + ","

                for j in range(len(pack_rigth_outs[0])):
                    aux_str_out += str(pack_rigth_outs[i][j]) + ","

                aux_str_in += "\n"
                aux_str_out += "\n\n"

                writing_string = aux_str_in + aux_str_out
                indx = np.argmax(pack_rigth_outs[i])
                addit_str[indx] += writing_string
                addit_cnt[indx] +=  1
                file.write(writing_string)

        for i in range(3):
            with open(ADDIT_PATHS[i], "w") as file:
                file.write(f"{addit_cnt[i]} \n\n")
                file.write(addit_str[i])
        
        return


    def learn(self, kind = 0, pack_data_in = [], pack_rigth_outs = [], write_file = False):
        
        if write_file:
            print("Writing file")
            with open(PATH_SETS, "w") as file:
                file.write(f"{len(pack_data_in)} \n\n")
                writing_string = ""

                for i in range(len(pack_data_in)):
                    aux_str_in = ""
                    aux_str_out = ""

                    for j in range(len(pack_data_in[0])):
                        aux_str_in += str(pack_data_in[i][j]) + ","

                    for j in range(len(pack_rigth_outs[0])):
                        aux_str_out += str(pack_rigth_outs[i][j]) + ","

                    aux_str_in += "\n"
                    aux_str_out += "\n\n"

                    writing_string += aux_str_in + aux_str_out
                file.write(writing_string)

        print("Training: ")
        if kind == 0:
            print("     All")
            train_set_name = SETS_NAME
        else:
            print("     "+ADDIT_SETS[kind-1])
            train_set_name = SETS_NAME + ADDIT_SETS[kind-1]

        for p in range(self.pop_size):
            self.handler.set_active_net(self.names[p])
            # self.handler.active_net_init_gradient(train_set_name)
            aux = self.handler.active_net_launch_gradient(iterations=GRADIENT_ITERATIONS, batch_size=netStandalone.FULL_BATCH, alpha=1, alpha_decay=0.001, reg_lambda=0.1, error_threshold=1, norm=netStandalone.NORM_2, file=train_set_name, file_reload=netStandalone.REUSE_FILE)
            for i in range(len(aux)):
                if np.isnan(aux[i]):
                    log("NaN en red "+str(p)+". Arch: "+str(self.archs[p][0])+", "+str(self.archs[p][1])+", "+str(self.archs[p][2])+", ","ERROR")
                    raise NameError("NaN detected in gradient")
        return


    def screen(self, pack_data_in, pack_rigth_outs, balance = False):

        self._pack_data_out = self.exe_pack_all(pack_data_in, self._data_out)

        for p in range(self.pop_size):
            self.scores[p] = 0

            if balance:
                for i in range(len(pack_data_in)):
                    self.scores[p] += np.sum(np.abs(pack_rigth_outs[i] - self._data_out[p][i])**2)
                # aux_score = np.zeros(3)
                # for i in range(len(pack_data_in)):
                #     aux_score += np.abs(pack_rigth_outs[i] - self._data_out[p][i])
                # aux_mult = 1
                # aux_score = 1/(1+aux_score*aux_score)
                # for s in range(len(aux_score)):
                #     aux_mult *= aux_score[s]
                # self.scores[p] = 1/aux_mult
            else:
                for i in range(len(pack_data_in)):
                    self.scores[p] += np.sum(np.abs(pack_rigth_outs[i] - self._data_out[p][i]))

        self.black_list = np.argsort(self.scores)

        return

    def evolve(self, survival_factor = 0.5):

        self.gen += 1
        init_p = int(self.pop_size*survival_factor) 
        best_index = self.black_list[0]

        for p in range(init_p,self.pop_size):
            index = self.black_list[p]
            self.handler.delete_net(self.names[index])
            
            self.archs[index] = [self.archs[best_index][0]+np.random.randint(2,5), self.archs[best_index][1]+np.random.randint(2,5), self.archs[best_index][2]]
            activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2_SOFT_MAX])
            self.names[index] = "AG_NET_G" + str(self.gen) +"_S" + str(best_index) + "_" + str(p-init_p)
            self.handler.net_create_random_from_vector(self.names[index], self.imp, self.n_ins, n_p_l=netStandalone.v_size_t(self.archs[index]),activation_type=activation_type)

            return

def gen_fig_examp(n_examples):
    
    imgs = np.zeros((n_examples,1000*1000*3))
    right_outs = np.zeros((n_examples,3))

    for i in tqdm(range(n_examples)):
        rnd = np.random.randint(1,4)
        if rnd==1:
            right_outs[i,rnd-1] = 1
            gen_square(i,imgs)
        if rnd==2:
            right_outs[i,rnd-1] = 1
            gen_triangle(i,imgs)
        if rnd==3:
            right_outs[i,rnd-1] = 1
            gen_circle(i,imgs)
    return imgs, right_outs

def gen_square(i,images):

    pos_x = np.random.randint(10,800)
    pos_y = np.random.randint(10,800)
    for x in range(100):
        for y in range(100):
            #Cuadrados
            images[i,pos_x + 1000*(y+pos_y) + x] = 200
            images[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
            images[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200
    return

def gen_triangle(i,images):

    pos_x = np.random.randint(10,800)
    pos_y = np.random.randint(10,800)
    for x in range(100):
        for y in range(100):
            #Triangulos
            if(y<x):
                images[i,pos_x + 1000*(y+pos_y) + x] = 200
                images[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
                images[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200
    return

def gen_circle(i,images):

    pos_x = np.random.randint(10,800)
    pos_y = np.random.randint(10,800)
    for x in range(100):
        for y in range(100):
            #Cuarto de circulo
            if(y<np.sqrt(100*100-x*x)):
                images[i,pos_x + 1000*(y+pos_y) + x] = 200
                images[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
                images[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200
    return
