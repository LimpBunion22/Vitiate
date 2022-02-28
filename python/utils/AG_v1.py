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

        self.pop_size = population_size
        self.names = []
        self.scores = np.zeros(self.pop_size)
        self.black_list = np.zeros(self.pop_size)

        self.handler = netStandalone.net_handler(PATH)
        for p in range(self.pop_size):
            # neurons = [np.random.randint(n_ins,n_ins +50), np.random.randint(int(n_ins/2),int(n_ins/2) + 50), np.random.randint(n_outs,n_outs + 50), n_outs]
            neurons = [np.random.randint(n_ins,n_ins +50), np.random.randint(n_outs,n_outs + 50), n_outs]
            self.names.append("AG_NET_G0_" + str(p))
            self.handler.net_create_random_from_vector(self.names[p], net_imp, n_ins, n_p_l=netStandalone.v_size_t(neurons))
        
        self._data_out = []    
        self._pack_data_out = []

        return


    def exe_all(self, data_in, data_out = []):

        while(len(data_out) < self.pop_size):
            data_out.append([])

        for p in range(self.pop_size):
            self.handler.set_active_net(self.names[p])
            data_out[p] = self.handler.active_net_launch_forward(netStandalone.v_data_type(data_in))

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
                pack_data_out[p][i] = self.handler.active_net_launch_forward(netStandalone.v_data_type(pack_data_in[i]))

        return pack_data_out


    def exe_best(self, data_in):

        self.handler.set_active_net(self.names[self.black_list[0]])

        return self.handler.active_net_launch_forward(netStandalone.v_data_type(data_in))

    def exe_pack_best(self, pack_data_in):

        pack_data_out = []
        self.handler.set_active_net(self.names[self.black_list[0]])

        for i in range(len(pack_data_in)):
            pack_data_out.append(self.handler.active_net_launch_forward(netStandalone.v_data_type(pack_data_in[i])))

        return pack_data_out


    def learn(self, pack_data_in, pack_rigth_outs):


        with open(PATH_SETS, "w") as file:
            file.write(f"{len(pack_data_in)} \n\n")

            for i in range(len(pack_data_in)):
                aux_str_in = ""
                aux_str_out = ""

                for j in range(len(pack_data_in[0])):
                    aux_str_in += str(pack_data_in[i][j]) + " "

                for j in range(len(pack_rigth_outs[0])):
                    aux_str_out += str(pack_rigth_outs[i][j]) + " "

                aux_str_in += "\n"
                aux_str_out += "\n\n"

                file.write(aux_str_in + aux_str_out)

        for p in range(self.pop_size):
            self.handler.set_active_net(self.names[p])
            self.handler.active_net_init_gradient(SETS_NAME)
            self.handler.active_net_launch_gradient(50, error_threshold = 0.01, multiplier = 2)
        
        return


    def screen(self, pack_data_in, pack_rigth_outs):

        self._pack_data_out = self.exe_pack_all(pack_data_in, self._data_out)

        for p in range(self.pop_size):
            self.scores[p] = 0

            for i in range(len(pack_data_in)):
                self.scores[p] += np.sum(np.abs(pack_rigth_outs[i] - self._data_out[p][i]))

        self.black_list = np.argsort(self.scores)

        return
                
        