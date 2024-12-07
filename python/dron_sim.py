
import numpy as np


class dron_simulator():


    def __init__(self):

        self.pos = 20000*(np.random.rand(3)-0.5)
        self.pos[2] = 5
        self.obj = 40000*(np.random.rand(3)-0.5)
        self.obj[2] = 0
        self.vel = (np.random.rand(3)-0.5)

        self.vampire = False
        self.vampire_pos = 20000*(np.random.rand(3)-0.5)
        self.vampire_pos[2] = 0
        self.vampire_vel = 10*np.ones(3)

        return
    
    def run(self, thrust):

        for i in range(3):
            thrust[i] = np.min(thrust[i],1)
            thrust[i] = np.max(thrust[i],-1)

        self.pos += 0.1*self.vel
        self.vel += 0.1*thrust

        dist = self.obj - self.pos
        if np.sqrt(np.sum(dist*dist)) < 5 :
            return 1,[]

        for i in range(3):
            self.vel[i] = np.min(self.vel[i],5)
            self.vel[i] = np.max(self.vel[i],-5)

        if (self.vampire):
            self.vampire_pos += 0.1*self.vampire_vel
            v_dist = self.obj - self.vampire_pos
            v_s_dist = np.sqrt(np.sum(v_dist*v_dist))

            if v_s_dist < 10:
                if v_s_dist < 5:
                    return -1,[]
                else:
                    self.vampire = False
                    self.vampire_pos = 20000*(np.random.rand(3)-0.5)
                    self.vampire_pos[2] = 0
                    self.vampire_vel = 10*np.ones(3)

            self.vampire_vel += 0.5*(self.obj - self.vampire_pos)/v_s_dist
            for i in range(3):
                self.vampire_vel[i] = np.min(self.vampire_vel[i],10)
                self.vampire_vel[i] = np.max(self.vampire_vel[i],-10)

        else:
            if(np.rand()>0.95):
                self.vampire = True

        
        res = np.concatenate((dist,))


