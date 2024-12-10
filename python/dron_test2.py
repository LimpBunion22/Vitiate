
from dron_sim import *
import numpy as np
import netStandalone
import os



# simulator = dron_simulator()        
# input = netStandalone.v_float(np.concatenate((simulator.obj-simulator.pos,simulator.vel,[-1],np.zeros(3))))
# thrust = [0,0,0]

# objetivo = (simulator.obj[0], simulator.obj[1], simulator.obj[2])
# ruta = 5000*[(simulator.pos[0], simulator.pos[1], simulator.pos[2])]
# vampires = []
# vamp = False

# iterations = 1
# while(True):
#     [status,response] = simulator.run(thrust)
#     ruta[iterations] = (simulator.pos[0], simulator.pos[1], simulator.pos[2])
#     if simulator.vampire==1:
#         if vamp:
#             vampires[-1].append((simulator.vampire_pos[0], simulator.vampire_pos[1], simulator.vampire_pos[2]))
#         else:
#             vampires.append([(simulator.vampire_pos[0], simulator.vampire_pos[1], simulator.vampire_pos[2])])
#             vamp = True
#     else:
#         vamp = False

#     if status == -1:
#         print("Destroyed by Vampire")
#         break
#     if status == 1:
#         print("Objective Achieved")
#         break
#     iterations += 1
#     if iterations >= 5000:
#         print("Timeout")
#         break
#     input = netStandalone.v_float(response)
#     n1 = np.max((np.linalg.norm(response[0:2]),0.1))
#     n2 = np.max((np.linalg.norm(response[3:5]),0.1))
#     thrust = 50*(response[0:3]/n1 - 0.1*response[3:6]/n2)

# dibujar_escena_3d(objetivo, ruta, vampires)


PATH = os.path.join(os.environ["HOME"], "workspace_development")
handler = netStandalone.handler(PATH)

name = "Sparrow"
handler.instantiate(name, netStandalone.GPU)
handler.set_active_net(name)

handler.set_input_size(10)
handler.build_fully_layer(60)
handler.build_fully_layer(30)
handler.build_fully_layer(3)
handler.build_net()

test_name = "Trinity"
print("Writing File")
generate_trainig_file(file_name = test_name, n_examples = 50)
print("File ready")

# handler.attr(netStandalone.EPOCHS, 10)\
#     .attr(netStandalone.BATCH_SIZE, 10000)\
#     .attr(netStandalone.ALPHA, 10.1)\
#     .attr(netStandalone.ALPHA_DECAY, 0.0001)\
#     .attr(netStandalone.ERROR_THRESHOLD, 0.001)\
#     .attr(netStandalone.ABS)\
#     .attr(netStandalone.ADAM)

# print("Training")
# prev = 1e17
# for i in range(5):
#     err = handler.run_gradient(test_name, netStandalone.REUSE_FILE)
#     print(err)
#     if err[-1]>prev:
#         break
#     prev = err[-1]
# print("Training completed")

# simulator = dron_simulator()        
# input = netStandalone.v_float(np.concatenate((simulator.obj-simulator.pos,simulator.vel,[-1],np.zeros(3))))
# thrust = handler.run_forward(input)

# objetivo = (simulator.obj[0], simulator.obj[1], simulator.obj[2])
# ruta = 10000*[(simulator.pos[0], simulator.pos[1], simulator.pos[2])]
# vampires = []
# vamp = False

# iterations = 1
# while(True):
#     [status,response] = simulator.run(thrust)
#     ruta[iterations] = (simulator.pos[0], simulator.pos[1], simulator.pos[2])
#     if simulator.vampire==1:
#         if vamp:
#             vampires[-1].append((simulator.vampire_pos[0], simulator.vampire_pos[1], simulator.vampire_pos[2]))
#         else:
#             vampires.append([(simulator.vampire_pos[0], simulator.vampire_pos[1], simulator.vampire_pos[2])])
#             vamp = True
#     else:
#         vamp = False

#     if status == -1:
#         print("Destroyed by Vampire")
#         break
#     if status == 1:
#         print("Objective Achieved")
#         break
#     iterations += 1
#     if iterations >= 10000:
#         print("Timeout")
#         break
#     input = netStandalone.v_float(response)
#     thrust = handler.run_forward(input)

# dibujar_escena_3d(objetivo, ruta, vampires)