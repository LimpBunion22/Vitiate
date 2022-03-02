from struct import pack
import netStandalone
import numpy as np
import copy
import os
from tqdm import tqdm

PATH = os.path.join(os.environ['HOME'], "workspace_development")
SETS_NAME = "_temporal_sets"
PATH_SETS = os.path.join(PATH, SETS_NAME + ".csv")

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
            self.archs.append([np.random.randint(5,150), np.random.randint(5,50), n_outs])
            activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2])
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


    def learn(self, pack_data_in, pack_rigth_outs, write_file = True):
        
        if write_file:
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

                    file.write(aux_str_in + aux_str_out)

        print("Training")
        for p in tqdm(range(self.pop_size)):
            self.handler.set_active_net(self.names[p])
            self.handler.active_net_init_gradient(SETS_NAME)
            self.handler.active_net_launch_gradient(2, error_threshold = 0.01, multiplier = 2)
        
        return


    def screen(self, pack_data_in, pack_rigth_outs):

        self._pack_data_out = self.exe_pack_all(pack_data_in, self._data_out)

        for p in range(self.pop_size):
            self.scores[p] = 0

            for i in range(len(pack_data_in)):
                self.scores[p] += np.sum(np.abs(pack_rigth_outs[i] - self._data_out[p][i]))

        self.black_list = np.argsort(self.scores)

        return

    def evolve(self, survival_factor = 0.5):

        self.gen += 1
        intit_p = int(self.pop_size*survival_factor) 

        for p in range(intit_p,self.pop_size):
            index = self.black_list[p]
            self.handler.delete_net(self.names[index])
            
            self.archs[index] = [self.archs[index][0]+np.random.randint(2,5), self.archs[index][1]+np.random.randint(2,5), self.archs[index][2]]
            activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2])
            self.names[index] = "AG_NET_G" + str(self.gen) +"_" + str(p)
            self.handler.net_create_random_from_vector(self.names[index], self.imp, self.n_ins, n_p_l=netStandalone.v_size_t(self.archs[p]),activation_type=activation_type)

            return
