import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import netStandalone
import os
from tqdm import tqdm

PATH = os.path.join(os.environ['HOME'], "workspace_development")
handler = netStandalone.net_handler(PATH)
activation_type=netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2, netStandalone.RELU2])
handler.net_create_random_from_vector("FPGA_net", netStandalone.FPGA, 1, netStandalone.v_size_t([1,1]), netStandalone.v_int([netStandalone.RELU2, netStandalone.RELU2]))
handler.net_create_random_from_vector("GPU_net", netStandalone.GPU, 1000000, netStandalone.v_size_t([256,64,3]),activation_type)
handler.set_active_net("FPGA_net")

TRAIN_PACK_SZ = 20
VAL_PACK_SZ = 50
train_imgs = np.zeros((3*TRAIN_PACK_SZ,1000*1000*3))
train_right_outs = np.zeros((3*TRAIN_PACK_SZ,3))
process_train_img = []
val_imgs = np.zeros((3*VAL_PACK_SZ,1000*1000*3))
val_right_outs = np.zeros((3*VAL_PACK_SZ,3))
process_val_img = []
val_net_outs = []

print("Generando imagenes")

for i in tqdm(range(TRAIN_PACK_SZ)):
    pos_x = np.random.randint(10,800)
    pos_y = np.random.randint(10,800)
    train_right_outs[i,:] = [10,-1,-1]
    train_right_outs[i+TRAIN_PACK_SZ,:] = [-1,10,-1]
    train_right_outs[i+2*TRAIN_PACK_SZ,:] = [-1,-1,10]
    for x in range(100):
        for y in range(100):
            #Cuadrados
            train_imgs[i,pos_x + 1000*(y+pos_y) + x] = 200
            train_imgs[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
            train_imgs[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200

            #Triangulos
            if(y<x):
                train_imgs[i,pos_x + 1000*(y+pos_y) + x] = 200
                train_imgs[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
                train_imgs[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200

            #Cuarto de circulo
            if(y>np.sqrt(100*100-x*x)):
                train_imgs[i,pos_x + 1000*(y+pos_y) + x] = 200
                train_imgs[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
                train_imgs[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200

for i in tqdm(range(VAL_PACK_SZ)):
    pos_x = np.random.randint(10,800)
    pos_y = np.random.randint(10,800)
    val_right_outs[i,:] = [10,-1,-1]
    val_right_outs[i+VAL_PACK_SZ,:] = [-1,10,-1]
    val_right_outs[i+2*VAL_PACK_SZ,:] = [-1,-1,10]
    for x in range(100):
        for y in range(100):
            #Cuadrados
            val_imgs[i,pos_x + 1000*(y+pos_y) + x] = 200
            val_imgs[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
            val_imgs[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200

            #Triangulos
            if(y<x):
                val_imgs[i,pos_x + 1000*(y+pos_y) + x] = 200
                val_imgs[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
                val_imgs[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200

            #Cuarto de circulo
            if(y>np.sqrt(100*100-x*x)):
                val_imgs[i,pos_x + 1000*(y+pos_y) + x] = 200
                val_imgs[i,1000000 + pos_x + 1000*(y+pos_y) + x] = 200
                val_imgs[i,2000000 + pos_x + 1000*(y+pos_y) + x] = 200

print("Processing images")
for i in tqdm(range(3*TRAIN_PACK_SZ)):
    handler.set_active_net("FPGA_net")
    process_train_img.append(handler.process_img_1000x1000(netStandalone.v_float(train_imgs[i])))

print("Wrinting file")
with open(os.path.join(PATH, "images_sets.csv"), "w") as file:
    file.write(f"{3*TRAIN_PACK_SZ} \n\n")

    for i in tqdm(range(3*TRAIN_PACK_SZ)):
        aux_str_in = ""
        aux_str_out = ""

        for j in range(len(process_train_img[0])):
            aux_str_in += str(process_train_img[i][j]) + ","

        for j in range(len(train_right_outs[0])):
            aux_str_out += str(train_right_outs[i][j]) + ","

        aux_str_in += "\n"
        aux_str_out += "\n\n"

        file.write(aux_str_in + aux_str_out)

print("Training net")
handler.set_active_net("GPU_net")
handler.active_net_init_gradient("images_sets")
handler.active_net_launch_gradient(250, error_threshold=0.1, multiplier=2)

print("Validating")
fig_ident_train = 0
handler.set_active_net("GPU_net")
train_net_outs = []
for i in tqdm(range(3*TRAIN_PACK_SZ)):
    train_net_outs.append(handler.active_net_launch_forward(process_train_img[i]))
    aux = np.argsort(train_net_outs[i])
    if aux[2] == int(i/VAL_PACK_SZ):
        fig_ident_train += 1

print("Figuras acertadas entrenamiento: " + str(fig_ident_train)+"/"+str(3*TRAIN_PACK_SZ))

fig_ident = 0
for i in tqdm(range(3*VAL_PACK_SZ)):
    handler.set_active_net("FPGA_net")
    process_val_img.append(handler.process_img_1000x1000(netStandalone.v_float(val_imgs[i])))
    handler.set_active_net("GPU_net")
    val_net_outs.append(handler.active_net_launch_forward(process_val_img[i]))
    aux = np.argsort(val_net_outs[i])
    if aux[2] == int(i/VAL_PACK_SZ):
        fig_ident += 1

print("Figuras acertadas validación: " + str(fig_ident)+"/"+str(3*VAL_PACK_SZ))

cnt = 0
while (fig_ident < 0.85*3*VAL_PACK_SZ) and (cnt < 20):
    cnt +=1
    print("Training net")
    handler.set_active_net("GPU_net")
    handler.active_net_init_gradient("images_sets")
    handler.active_net_launch_gradient(250, error_threshold=0.1, multiplier=2)

    print("Validating")
    fig_ident_train = 0
    handler.set_active_net("GPU_net")
    train_net_outs = []
    for i in tqdm(range(3*TRAIN_PACK_SZ)):
        train_net_outs.append(handler.active_net_launch_forward(process_train_img[i]))
        aux = np.argsort(train_net_outs[i])
        if aux[2] == int(i/VAL_PACK_SZ):
            fig_ident_train += 1

    print("Figuras acertadas entrenamiento: " + str(fig_ident_train)+"/"+str(3*TRAIN_PACK_SZ))

    fig_ident = 0
    for i in tqdm(range(3*VAL_PACK_SZ)):
        handler.set_active_net("FPGA_net")
        process_val_img.append(handler.process_img_1000x1000(netStandalone.v_float(val_imgs[i])))
        handler.set_active_net("GPU_net")
        val_net_outs.append(handler.active_net_launch_forward(process_val_img[i]))
        aux = np.argsort(val_net_outs[i])
        if aux[2] == int(i/VAL_PACK_SZ):
            fig_ident += 1

    print("Figuras acertadas validación: " + str(fig_ident)+"/"+str(3*VAL_PACK_SZ))
# for x in range(1000):
#     for y in range(1000):
#         or_img[y,x,0] = original_image[1000*y + x]
#         or_img[y,x,1] = original_image[1000000 + 1000*y + x]
#         or_img[y,x,2] = original_image[2000000 + 1000*y + x]

#         f_img[y,x,0] = int(filtered_image[1000*y + x])
#         f_img[y,x,1] = int(filtered_image[1000*y + x])
#         f_img[y,x,2] = int(filtered_image[1000*y + x])

# # create figure
# fig = plt.figure(figsize=(10, 7))
  
# # setting values to rows and column variables
# rows = 1
# columns = 2

# # Adds a subplot at the 1st position
# fig.add_subplot(rows, columns, 1)
  
# # showing image
# plt.imshow(or_img)
# plt.axis('off')
# plt.title("Original")

# # Adds a subplot at the 2nd position
# fig.add_subplot(rows, columns, 2)
  
# # showing image
# plt.imshow(f_img)
# plt.axis('off')
# plt.title("Filtered")

# plt.show(block=True)
# # input("Press any key to continue.")


