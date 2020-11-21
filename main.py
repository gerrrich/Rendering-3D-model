import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img


def R(ang, ax):
    c = np.cos(ang * np.pi / 180)
    s = np.sin(ang * np.pi / 180)
    if ax == 'x':
        return np.array([[1, 0, 0, 0],
                         [0, c, -s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])
    if ax == 'y':
        return np.array([[c, 0, s, 0],
                         [0, 1, 0, 0],
                         [-s, 0, c, 0],
                         [0, 0, 0, 1]])
    if ax == 'z':
        return np.array([[c, -s, 0, 0],
                         [s, c, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])


def S(a, b, c):
    return np.array([[a, 0, 0, 0],
                     [0, b, 0, 0],
                     [0, 0, c, 0],
                     [0, 0, 0, 1]])


def T(a, b, c):
    return np.array([[1, 0, 0, a],
                     [0, 1, 0, b],
                     [0, 0, 1, c],
                     [0, 0, 0, 1]])


def Mw2c(v0, v1):
    cd = norm(v0 - v1)
    up = np.array([0, 1, 0])
    cr = norm(np.cross(up, cd))
    cu = np.cross(cd, cr)
    L = np.array([[cr[0], cr[1], cr[2], 0],
                  [cu[0], cu[1], cu[2], 0],
                  [cd[0], cd[1], cd[2], 0],
                  [0, 0, 0, 1]])

    R = np.array([[1, 0, 0, -v0[0]],
                  [0, 1, 0, -v0[1]],
                  [0, 0, 1, -v0[2]],
                  [0, 0, 0, 1]])

    return L.dot(R)


def Mproj_o(l, r, b, t, n, f):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)],
                     [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, 2 / (f - n), -(f + n) / (f - n)],
                     [0, 0, 0, 1]])


def Mproj_p(l, r, b, t, n, f):
    return np.array([[2 * n / (r - l), 0, (r + l) / (r - l), 0],
                     [0, 2 * n / (t - b), (t + b) / (t - b), 0],
                     [0, 0, (f + n) / (n - f), 2 * n * f / (n - f)],
                     [0, 0, -1, 0]])


def Mviewport(x=0, y=0, w=1024, h=1024):
    return np.array([[w / 2, 0, 0, x + w / 2],
                     [0, h / 2, 0, y + h / 2],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])


def norm(v):
    no = 0
    for i in v:
        no += i ** 2
    no = np.sqrt(no)
    if no == 0:
        return v
    return v / no


def back_face_culling(p, v0, v1, v2, n=0):
    v1 = np.array([v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]])
    v2 = np.array([v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]])
    v0 = np.array([v0[0] - p[0], v0[1] - p[1], v0[2] - p[2]])
    nn = norm(np.cross(v1, v2))
    s = np.inner(np.array([0, 0, -1]), nn)
    # s = np.inner(v0, nn)
    return s >= 0


def get_barycentric_coords(p, v0, v1, v2):
    aa0 = p[0] - v0[0]
    aa1 = p[1] - v0[1]
    bb0 = v1[0] - v0[0]
    bb1 = v1[1] - v0[1]
    cc0 = v2[0] - v0[0]
    cc1 = v2[1] - v0[1]

    c = (aa0 * bb1 - aa1 * bb0) / (cc0 * bb1 - cc1 * bb0)
    b = (aa1 - c * cc1) / bb1
    a = 1 - b - c
    return a, b, c


with open('obj/african_head/african_head.obj') as file:
    obj = file.readlines()

fig = plt.figure(figsize=(10, 10))
texture = img.imread('obj/african_head/african_head_diffuse.tga')
vertices = []
texture_v = []
normals_v = []
new_normals_v = []
faces = []
new_vertices = []
image = np.zeros((1024, 1024, 3), dtype=np.uint8)
color = np.array([0, 255, 0], dtype=np.uint8)
camera = np.array([0, 0, 0])
z_buffer = np.ones((1024, 1024), dtype=np.float)

for line in obj:
    temp = line[:-1].split()
    if len(temp) == 0:
        continue
    elif temp[0] == 'v':
        vertices.append(np.array([float(temp[1]), float(temp[2]), float(temp[3]), 1]))
    elif temp[0] == 'vt':
        texture_v.append([float(temp[1]), float(temp[2])])
    elif temp[0] == 'vn':
        normals_v.append(np.array([float(temp[1]), float(temp[2]), float(temp[3]), 1]))
    elif temp[0] == 'f':
        cur1 = temp[1].split('/')
        cur2 = temp[2].split('/')
        cur3 = temp[3].split('/')
        faces.append([[int(cur1[0]) - 1, int(cur1[1]) - 1, int(cur1[2]) - 1],
                      [int(cur2[0]) - 1, int(cur2[1]) - 1, int(cur2[2]) - 1],
                      [int(cur3[0]) - 1, int(cur3[1]) - 1, int(cur3[2]) - 1]])
Mo2w = R(0, 'z').dot(R(0, 'y').dot(R(0, 'x').dot(T(0, 0, 0).dot(S(0.8, 0.8, 0.8)))))
# M = Mw2c(np.array([0, 0, 0]), np.array([0, 0, 0])).dot(Mo2w)
# print(Mw2c(0, 0, 0, -4, -4, -2).dot(T(-2, -2, -2).dot(np.array([2, 2, 2, 1]))))

# MT = Mw2c(2, 2, 2, -2, -2, 0).T.dot(
#     R(15, 'z').T.dot(R(280, 'y').T.dot(R(5, 'x').T.dot(T(-1, 0, -2).T.dot(S(0.8, 0.8, 0.8).T)))))

for i in range(len(vertices)):
    new_vertices.append(vertices[i])#.dot(vertices[i]))
# for i in range(len(normals_v)):
#     new_normals_v.append(MT.dot(normals_v[i]))

ll = min(new_vertices, key=lambda x: x[0])
rr = max(new_vertices, key=lambda x: x[0])
bb = min(new_vertices, key=lambda x: x[1])
tt = max(new_vertices, key=lambda x: x[1])
nn = min(new_vertices, key=lambda x: x[2])
ff = max(new_vertices, key=lambda x: x[2])

l = ll[0] - (rr[0] - ll[0]) / 20
r = rr[0] + (rr[0] - ll[0]) / 20
b = bb[1] - (tt[1] - bb[1]) / 20
t = tt[1] + (tt[1] - bb[1]) / 20
n = nn[2] - (ff[2] - nn[2]) / 20
f = ff[2] + (ff[2] - nn[2]) / 20

M = Mviewport().dot(Mproj_o(l, r, b, t, n, f))

# for i in range(len(new_normals_v)):
#     new_normals_v[i] = MT.dot(new_normals_v[i])

for face in faces:
    v1 = new_vertices[face[0][0]]
    v2 = new_vertices[face[1][0]]
    v3 = new_vertices[face[2][0]]
    # vn1 = new_normals_v[face[0][2]]
    # vn2 = new_normals_v[face[1][2]]
    # vn3 = new_normals_v[face[2][2]]

    #if back_face_culling(camera, v1, v2, v3):
    #    continue

    v1 = M.dot(v1)
    v2 = M.dot(v2)
    v3 = M.dot(v3)

    x_min = int(np.floor(min(v1, v2, v3, key=lambda x: x[0])[0]))
    x_max = int(np.ceil(max(v1, v2, v3, key=lambda x: x[0])[0]))
    y_min = int(np.floor(min(v1, v2, v3, key=lambda x: x[1])[1]))
    y_max = int(np.ceil(max(v1, v2, v3, key=lambda x: x[1])[1]))

    for i in range(x_min, x_max + 1):
        for j in range(y_min, y_max + 1):
            a, b, c = get_barycentric_coords([i, j], v1, v2, v3)
            if a >= 0 and b >= 0 and c >= 0:
                z = a * v1[2] + b * v2[2] + c * v3[2]
                if z < z_buffer[1024 - j, i]:
                    z_buffer[1024 - j, i] = z
                    u = a * texture_v[face[0][1]][0] + b * texture_v[face[1][1]][0] + c * texture_v[face[2][1]][0]
                    v = a * texture_v[face[0][1]][1] + b * texture_v[face[1][1]][1] + c * texture_v[face[2][1]][1]
                    image[1024 - j][i] = texture[1024 - round(v * len(texture))][round(u * len(texture[0]))]

plt.imshow(image)
plt.show()
