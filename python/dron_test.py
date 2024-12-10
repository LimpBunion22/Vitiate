import numpy as np
import netStandalone
import os
from dron_sim import dron_simulator,dibujar_escena_3d
from tqdm import tqdm


test_name = "Trinity"
PATH = os.path.join(os.environ["HOME"], "workspace_development")
handler = netStandalone.handler(PATH)


POPULATION = 5
GENERATIONS = 20
SIMS = 5

name_list = []
saved_batch = int(0.3*POPULATION)
clone_batch = int(0.2*POPULATION)
sons_n = int(clone_batch/saved_batch)
clone_batch = sons_n*saved_batch
new_batch = POPULATION - saved_batch - clone_batch
seed_cnt = POPULATION

for p in range (POPULATION):
    name = "S"+str(p)
    name_list.append(name)
    handler.instantiate(name, netStandalone.CPU)
    handler.set_active_net(name)

    handler.set_input_size(10)
    handler.build_fully_layer(np.random.randint(5,55))
    handler.build_fully_layer(np.random.randint(5,25))
    handler.build_fully_layer(3)
    handler.build_net()

print_cnt = 0
for g in tqdm(range(GENERATIONS)):

    # interrupt_lim = 150 + int(200/(g+1))

    scores = np.zeros(POPULATION)
    handler.configure_gradient_workload(POPULATION, test_name)
    for p in range(POPULATION):
        handler.set_active_net(name_list[p])
        handler.attr(netStandalone.EPOCHS, 1)\
            .attr(netStandalone.BATCH_SIZE, 10000)\
            .attr(netStandalone.ALPHA, 1.1)\
            .attr(netStandalone.ALPHA_DECAY, 0.0001)\
            .attr(netStandalone.ERROR_THRESHOLD, 0.001)\
            .attr(netStandalone.ABS)\
            .attr(netStandalone.ADAM)

        handler.enqueue_gradient(name_list[p])
        # err = handler.run_gradient(test_name, netStandalone.REUSE_FILE)
        # scores[p] = -err[-1]


        # for s in range(SIMS):
        #     simulator = dron_simulator()        
        #     input = netStandalone.v_float(np.concatenate((simulator.obj-simulator.pos,[-1],np.zeros(3))))
        #     thrust = handler.run_forward(input)

        #     original_dist = np.linalg.norm(simulator.obj - simulator.pos)
        #     OR_dist = original_dist

        #     iterations = 1
        #     while(True):
        #         [status,response] = simulator.run(thrust)
        #         if status == -1:
        #             scores[p] -= 1000
        #             scores[p] += 10*(OR_dist - simulator.s_dist)
        #             break
        #         if status == 1:
        #             scores[p] += 50000
        #             break
        #         iterations += 1
        #         scores[p] -= 0.5/interrupt_lim
        #         if iterations >= interrupt_lim:
        #             if simulator.s_dist > original_dist:
        #                 # scores[p] -= 10*simulator.s_dist
        #                 scores[p] += 10*(OR_dist - simulator.s_dist)
        #                 break
        #             else:
        #                 original_dist = simulator.s_dist
        #                 scores[p] += original_dist - simulator.s_dist
        #                 iterations = 0
        #         input = netStandalone.v_float(response)
        #         thrust = handler.run_forward(input)
    
    results = handler.get_gradient_worload_results()
    for p in range(POPULATION):
        scores[p] = results[p][1]

    sorted_index = np.argsort(scores)
    for p in range(POPULATION-saved_batch):
        handler.delete_net(name_list[sorted_index[p]])

    for p in range(saved_batch):
        father_index = sorted_index[POPULATION-saved_batch+p]
        for s in range(sons_n):            
            son_index = sorted_index[new_batch+p*sons_n+s]
            name = name_list[father_index] + "_G"+str(g)+"M"+str(s)
            name_list[son_index] = name
            handler.clone(name_list[father_index],name)
            handler.set_active_net(name)
            handler.mutate(np.random.rand()-0.5)
    
    for p in range(new_batch):

        name = "S"+str(seed_cnt)
        seed_cnt += 1
        name_list[sorted_index[p]] = name
        handler.instantiate(name, netStandalone.CPU)
        handler.set_active_net(name)

        handler.set_input_size(10)
        handler.build_fully_layer(np.random.randint(5,55))
        handler.build_fully_layer(np.random.randint(5,25))
        handler.build_fully_layer(3)
        handler.build_net()

    print_cnt += 1
    if print_cnt==10:
        print_cnt = 0

        print("\n\nGEN "+str(g))
        for p in range(POPULATION-saved_batch,POPULATION):
            index = sorted_index[p]
            print("     "+name_list[index]+":   "+str(scores[index]))
        
        handler.set_active_net(name_list[sorted_index[POPULATION-1]])
        simulator = dron_simulator()        
        input = netStandalone.v_float(np.concatenate((simulator.obj-simulator.pos,simulator.vel,[-1],np.zeros(3))))
        thrust = handler.run_forward(input)

        objetivo = (simulator.obj[0], simulator.obj[1], simulator.obj[2])
        ruta = 5000*[(simulator.pos[0], simulator.pos[1], simulator.pos[2])]
        vampires = []
        vamp = False

        iterations = 1
        while(True):
            [status,response] = simulator.run(thrust)
            ruta[iterations] = (simulator.pos[0], simulator.pos[1], simulator.pos[2])
            if simulator.vampire==1:
                if vamp:
                    vampires[-1].append((simulator.vampire_pos[0], simulator.vampire_pos[1], simulator.vampire_pos[2]))
                else:
                    vampires.append([(simulator.vampire_pos[0], simulator.vampire_pos[1], simulator.vampire_pos[2])])
                    vamp = True
            else:
                vamp = False

            if status == -1:
                print("Destroyed by Vampire")
                break
            if status == 1:
                print("Objective Achieved")
                break
            iterations += 1
            if iterations >= 5000:
                print("Timeout")
                break
            input = netStandalone.v_float(response)
            thrust = handler.run_forward(input)

        dibujar_escena_3d(objetivo, ruta, vampires)

    
        
            

