from functions import *

import matplotlib.pyplot as plt

l2w = LocalToWorld(5, 10, 15, [-1, 0, -2], [0.8, 0.8, 0.8])
w2c = WorldToCamera([2, 2, 2], [-2, -2, 0])

image = get_texture_image('obj/african_head/african_head.obj', 'obj/african_head/african_head_diffuse.tga',
                          1024, 1024, 600, 800, l2w, w2c, 100, 300)

fig = plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()
