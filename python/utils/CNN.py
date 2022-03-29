
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import netStandalone
import os
from tqdm import tqdm
import AG_v1

PATH = os.path.join(os.environ['HOME'], "workspace_development")
handler = netStandalone.net_handler(PATH)
activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2])
handler.net_create_random_from_vector("FPGA_net", netStandalone.FPGA, 1, netStandalone.v_size_t([1,1]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2]))
my_ag = AG_v1.ag_handler(population_size = 12, n_ins = 10000, n_outs = 3, net_imp = netStandalone.CPU)
handler.set_active_net("FPGA_net")

TRAIN_PACK_SZ = 100
VAL_PACK_SZ = 50
process_train_img = []
val_imgs = np.zeros((3*VAL_PACK_SZ,1000*1000*3))
val_right_outs = np.zeros((3*VAL_PACK_SZ,3))
process_val_img = []
val_net_outs = []

print("Generando imagenes")
[train_imgs,train_right_outs] = AG_v1.gen_fig_examp(3*TRAIN_PACK_SZ)
[val_imgs,val_right_outs] = AG_v1.gen_fig_examp(3*VAL_PACK_SZ)

print("Processing images")
handler.set_active_net("FPGA_net")
for i in tqdm(range(3*TRAIN_PACK_SZ)):
    process_train_img.append(handler.process_img_1000x1000(netStandalone.v_float(train_imgs[i]), True))

for i in tqdm(range(3*VAL_PACK_SZ)):
    process_val_img.append(handler.process_img_1000x1000(netStandalone.v_float(val_imgs[i]), True))

print("Ploting sample")
or_img = np.zeros((1000,1000,3))
f_img = np.zeros((100,100,3))
for x in range(1000):
    for y in range(1000):
        or_img[y,x,0] = train_imgs[0][1000*y + x]
        or_img[y,x,1] = train_imgs[0][1000000 + 1000*y + x]
        or_img[y,x,2] = train_imgs[0][2000000 + 1000*y + x]
for x in range(100):
    for y in range(100):
        f_img[y,x,0] = int(process_train_img[0][100*y + x])
        f_img[y,x,1] = int(process_train_img[0][100*y + x])
        f_img[y,x,2] = int(process_train_img[0][100*y + x])
# create figure
fig = plt.figure(figsize=(10, 7))
rows = 1
columns = 2

fig.add_subplot(rows, columns, 1)
plt.imshow(or_img)
plt.axis('off')
plt.title("Original")

fig.add_subplot(rows, columns, 2)
plt.imshow(f_img)
plt.axis('off')
plt.title("Filtered")
plt.show(block=True)
# input("Press any key to continue.")

print("Writing files")
# my_ag.learn(process_train_img,train_right_outs,True)
my_ag.write_training_file( process_train_img, train_right_outs)
my_ag.learn(np.random.randint(0,4))

cnt = 0
fig_ident = 0
while (np.sum(fig_ident) < 0.85*3*VAL_PACK_SZ) and (cnt < 30):
    cnt +=1
    print("ITERACIÓN: "+str(cnt)+"\n")
    print("Validating")
    my_ag.screen(process_train_img, train_right_outs, True)
    print("RED: "+my_ag.names[my_ag.black_list[0]]+"\n")

    fig_ident = [0,0,0]
    total_fig = [0,0,0]
    train_net_outs = my_ag.exe_pack_best(process_train_img)
    print("Figuras acertadas entrenamiento:\n")
    for i in range(3*TRAIN_PACK_SZ):
        aux = np.argmax(train_net_outs[i])
        aux_2 = np.argmax(train_right_outs[i])
        total_fig[aux_2] += 1
        if aux == aux_2:
            fig_ident[aux_2] += 1
    
    print("     Cuadrados: " + str(fig_ident[0])+"/"+str(total_fig[0]))
    print("     Triángulos: " + str(fig_ident[1])+"/"+str(total_fig[1]))
    print("     Cuartos de esfera: " + str(fig_ident[2])+"/"+str(total_fig[2]))

    fig_ident = [0,0,0]  
    total_fig = [0,0,0] 
    val_net_outs = my_ag.exe_pack_best(process_val_img)
    print("Figuras acertadas validación:\n")
    for i in range(3*VAL_PACK_SZ):
        aux = np.argmax(val_net_outs[i])
        aux_2 = np.argmax(val_right_outs[i])
        total_fig[aux_2] += 1
        if aux == aux_2:
            fig_ident[aux_2] += 1
    
    print("     Cuadrados: " + str(fig_ident[0])+"/"+str(total_fig[0]))
    print("     Triángulos: " + str(fig_ident[1])+"/"+str(total_fig[1]))
    print("     Cuartos de esfera: " + str(fig_ident[2])+"/"+str(total_fig[2])) 

    my_ag.learn(np.random.randint(0,4))#np.argmin(fig_ident)+1
    my_ag.evolve(survival_factor=0.75)
    print("\n\n")

# # for x in range(1000):
# #     for y in range(1000):
# #         or_img[y,x,0] = original_image[1000*y + x]
# #         or_img[y,x,1] = original_image[1000000 + 1000*y + x]
# #         or_img[y,x,2] = original_image[2000000 + 1000*y + x]

# #         f_img[y,x,0] = int(filtered_image[1000*y + x])
# #         f_img[y,x,1] = int(filtered_image[1000*y + x])
# #         f_img[y,x,2] = int(filtered_image[1000*y + x])

# # # create figure
# # fig = plt.figure(figsize=(10, 7))
  
# # # setting values to rows and column variables
# # rows = 1
# # columns = 2

# # # Adds a subplot at the 1st position
# # fig.add_subplot(rows, columns, 1)
  
# # # showing image
# # plt.imshow(or_img)
# # plt.axis('off')
# # plt.title("Original")

# # # Adds a subplot at the 2nd position
# # fig.add_subplot(rows, columns, 2)
  
# # # showing image
# # plt.imshow(f_img)
# # plt.axis('off')
# # plt.title("Filtered")

# # plt.show(block=True)
# # # input("Press any key to continue.")


