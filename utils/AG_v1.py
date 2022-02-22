import netStandalone
import numpy as np

class ag_handler:


    def __init__(self, population_size, n_ins, n_outs, net_imp = netStandalone.CPU):

        self.pop_size = population_size
        self.names = []
        self.scores = np.zeros(self.pop_size)
        self.black_list = np.zeros(self.pop_size)

        self.handler = netStandalone.net_handler("/home/hai/workspace_development")
        for p in range(self.pop_size):
            neurons = [np.random.randint(n_ins,n_ins +50), np.random.randint(int(n_ins/2),int(n_ins/2) + 50), np.random.randint(n_outs,n_outs + 50), n_outs]
            self.names.append("AG_NET_G0_" + str(p))
            self.handler.net_create_random_from_vector(self.names[p], net_imp, n_ins, n_p_l=netStandalone.v_size_t(neurons))
        
        self._data_out = []        
        return

    def exe_all(self, data_in, data_out = []):

        while(len(data_out) < self.pop_size):
            data_out.append([])

        for p in range(self.pop_size):
            self.handler.set_active_net(self.names[p])
            data_out[p] = self.handler.active_net_launch_forward(data_in)
        return data_out

    def exe_best(self, data_in):

        self.handler.set_active_net(self.names[self.black_list[0]])
        return self.handler.active_net_launch_forward(data_in)

    def _screen(self, data_in, rigth_out):

        self._data_out = self.exe_all(data_in, self._data_out)

        for p in range(self.pop_size):
            self.scores[p] = -np.sum(np.abs(rigth_out - self._data_out[p]))

        self.black_list = np.argsort(self.scores)

        return

    def _mutate(self, n_saved, n_mutants):

        if n_saved*(n_mutants+1)> self.pop_size:
            print("Not enough pop")
            return

        for s in range(n_saved):
            for m in range(n_mutants):
                
        