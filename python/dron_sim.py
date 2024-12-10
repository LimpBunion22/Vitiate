
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # o 'Qt5Agg', dependiendo de su instalación
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import copy
from tqdm import tqdm

PATH = os.path.join(os.environ["HOME"], "workspace_development")

class dron_simulator():


    def __init__(self):

        self.pos = 1000*2*(np.random.rand(3)-0.5)
        self.pos[2] = 5
        self.obj = 1000*2*(np.random.rand(3)-0.5)
        self.obj[2] = 0
        self.vel = (np.random.rand(3)-0.5)

        self.vampire = -1
        self.vampire_pos = np.zeros(3)
        self.vampire_pos[2] = 0
        self.vampire_vel = 10*np.ones(3)

        return
    
    def run(self, c_thrust):
        thrust = np.clip(c_thrust, -1, 1)

        self.pos += 0.1*self.vel
        self.vel += 0.1*thrust

        dist = self.obj - self.pos
        self.s_dist = np.linalg.norm(dist)
        if self.s_dist < 50 :
            return 1,[]

        self.vel = np.clip(self.vel, -5, 5)

        v_dist = np.zeros(3)
        if (self.vampire == 1):
            self.vampire_pos += 0.1*self.vampire_vel
            v_dist = self.pos - self.vampire_pos
            v_s_dist = np.linalg.norm(v_dist)
            if v_s_dist < 10:
                if v_s_dist < 5:
                    return -1,[]
                else:
                    self.vampire = -1
                    self.vampire_pos = 2*1000*2*(np.random.rand(3)-0.5)
                    self.vampire_pos[2] = 0
                    self.vampire_vel = 10*np.ones(3)

            self.vampire_vel += 0.5*v_dist/v_s_dist
            self.vampire_vel = np.clip(self.vampire_vel, -10, 10)


        else:
            if(np.random.rand()>0.9999):
                self.vampire = 1

        
        res = np.concatenate((dist,self.vel,[self.vampire],v_dist))
        return 0,res
    
def write_training_file(file_name, pack_data_in, pack_rigth_outs):        

    print("Writing file")
    file_path = os.path.join(PATH, file_name+".csv")
    with open(file_path, "w") as file:
        file.write(f"\n{len(pack_data_in)}\n\n\n")
        file.write(f"{len(pack_rigth_outs[0])},{len(pack_data_in[0])}\n\n\n")

        aux_str_in = ""
        aux_str_out = ""
        aux_str_lab = ""
        for i in range(len(pack_data_in)):
            for j in range(len(pack_data_in[0])):
                aux_str_in += str(pack_data_in[i][j]) + ","
            for j in range(len(pack_rigth_outs[0])):
                aux_str_out += str(pack_rigth_outs[i][j]) + ","
            aux_str_lab += "0,"

        aux_str_in += "\n\n"
        aux_str_out += "\n\n"

        writing_string = aux_str_in + aux_str_out + aux_str_lab
        file.write(writing_string)    
    return
    
def generate_trainig_file(file_name, n_examples):

    pack_data_in = []
    pack_rigth_outs = []
    for ex in tqdm(range(n_examples)):
        simulator = dron_simulator()
        [status,response] = simulator.run([0,0,0])
        while(True):
            if status!= 0:
                break
            n1 = np.max((np.linalg.norm(response[0:2]),0.1))
            n2 = np.max((np.linalg.norm(response[3:5]),0.1))
            thrust = 50*(response[0:3]/n1 - 0.1*response[3:6]/n2)
            pack_data_in.append(copy.copy(response))
            pack_rigth_outs.append(copy.copy(thrust))
            [status,response] = simulator.run(thrust)
    write_training_file(file_name, pack_data_in, pack_rigth_outs)
    return

def dibujar_escena_3d(objetivo, ruta, vampires):
    # Crear la figura 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Dibujar el punto objetivo
    # objetivo = (x_obj, y_obj, z_obj)
    ax.scatter(objetivo[0], objetivo[1], objetivo[2], c='red', marker='o', s=50, label='Objetivo')
    
    # Dibujar la ruta
    # La ruta es una lista de puntos 3D
    # Primer punto: Origen (con una marca y color distinto)
    origen = ruta[0]
    ax.scatter(origen[0], origen[1], origen[2], c='green', marker='^', s=50, label='Origen')
    
    if len(ruta) > 1:
        # Dibujar el resto de la línea que va del origen al último punto
        ruta_np = np.array(ruta)
        # Del segundo punto en adelante formamos la línea
        ax.plot(ruta_np[:,0], ruta_np[:,1], ruta_np[:,2], c='blue', label='Ruta')
    
    # Dibujar los vampires
    # vampires es una lista de listas de puntos
    # Cada lista interna se dibuja como una línea separada
    colors = plt.cm.tab10(np.linspace(0, 1, len(vampires)))  # Genera colores distintos
    
    for i, v_line in enumerate(vampires, start=1):
        v_line_np = np.array(v_line)
        etiqueta = f"VAMPIRE {i}"
        ax.plot(v_line_np[:,0], v_line_np[:,1], v_line_np[:,2], color=colors[i-1], label=etiqueta)
    
    # Ajustar leyenda
    ax.legend()
    
    # Ajustar ejes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Mostrar el gráfico
    plt.show()


# Ejemplo de uso:
# objetivo = (10, 10, 10)
# ruta = [(0,0,0), (1,2,3), (2,4,6), (3,6,9)]
# vampires = [
#     [(2,2,2), (2,3,3), (2,4,5)],
#     [(4,4,4), (5,5,7), (6,6,10)]
# ]
# dibujar_escena_3d(objetivo, ruta, vampires)



