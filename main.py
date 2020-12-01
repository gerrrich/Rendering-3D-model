from functions import *

import matplotlib.pyplot as plt

l2w = LocalToWorld(5, 10, 15, [-1, 0, -2], [0.8, 0.8, 0.8])
w2c = WorldToCamera([2, 2, 2], [-2, -2, 0])

model = Model('obj/african_head/african_head.obj', 'obj/african_head/african_head_diffuse.tga', np.array([4, 4, 3, 1]))
model.create_camera_coordinates(l2w, w2c)

image1 = get_wire_image(1024, 1024, 500, 500, 8, 16 + 500, model)
image2 = get_face_image(1024, 1024, 500, 500, 16 + 500, 16 + 500, model)
image3 = get_texture_image(1024, 1024, 500, 500, 8, 8, model)
image4 = get_texture_image_with_with_light(1024, 1024, 500, 500, 16 + 500, 8, model)
image = np.zeros((1024, 1024, 3), dtype=np.uint8)

for i in range(1024):
    for j in range(1024):
        if i < 512 and j < 512:
            image[j][i] = image1[j][i]
        elif i >= 512 and j < 512:
            image[j][i] = image2[j][i]
        elif i < 512 and j >= 512:
            image[j][i] = image3[j][i]
        elif i >= 512 and j >= 512:
            image[j][i] = image4[j][i]

fig = plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()