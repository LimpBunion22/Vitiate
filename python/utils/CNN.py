import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import netStandalone
import os

PATH = os.path.join(os.environ['HOME'], "workspace_development")
handler = netStandalone.net_handler(PATH)
handler.net_create_random_from_vector("FPGA_net", netStandalone.FPGA, 1, n_p_l=netStandalone.v_size_t([1,1]))
handler.set_active_net("FPGA_net")

original_image = np.zeros(1000*1000*3)
or_img = np.zeros((1000,1000,3))
f_img = np.zeros((1000,1000,3))

for x in range(100):
    for y in range(100):
        original_image[100 + 1000*(y+100) + x] = 200
        original_image[1000000 + 100 + 1000*(y+100) + x] = 200
        original_image[2000000 + 100 + 1000*(y+100) + x] = 200

filtered_image = handler.process_img_1000x1000(netStandalone.v_data_type(original_image))

for x in range(1000):
    for y in range(1000):
        or_img[y,x,0] = original_image[1000*y + x]
        or_img[y,x,1] = original_image[1000000 + 1000*y + x]
        or_img[y,x,2] = original_image[2000000 + 1000*y + x]

        f_img[y,x,0] = int(filtered_image[1000*y + x])
        f_img[y,x,1] = int(filtered_image[1000*y + x])
        f_img[y,x,2] = int(filtered_image[1000*y + x])

# create figure
fig = plt.figure(figsize=(10, 7))
  
# setting values to rows and column variables
rows = 1
columns = 2

# Adds a subplot at the 1st position
fig.add_subplot(rows, columns, 1)
  
# showing image
plt.imshow(or_img)
plt.axis('off')
plt.title("Original")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)
  
# showing image
plt.imshow(f_img)
plt.axis('off')
plt.title("Filtered")

plt.show(block=True)
# input("Press any key to continue.")


