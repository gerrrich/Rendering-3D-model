import numpy as np
import matplotlib.image as img


def draw_line(x0, y0, x1, y1, image, color, location_x, location_y, view_height):
    sign_x = np.sign(x1 - x0)
    sign_y = np.sign(y1 - y0)

    d_x = abs(x1 - x0)
    d_y = abs(y1 - y0)

    if d_x > d_y:
        d = d_x
        dd = d_y
    else:
        d = d_y
        dd = d_x

    x_cur, y_cur = x0, y0
    error = d / 2
    image[view_height - y_cur, x_cur] = color

    for i in range(d):
        error -= dd
        if error < 0:
            error += d
            x_cur += sign_x
            y_cur += sign_y
        else:
            if d_x > d_y:
                x_cur += sign_x
            else:
                y_cur += sign_y
        image[view_height - y_cur, x_cur] = color


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


def S(v):
    return np.array([[v[0], 0, 0, 0],
                     [0, v[1], 0, 0],
                     [0, 0, v[2], 0],
                     [0, 0, 0, 1]])


def T(v):
    return np.array([[1, 0, 0, v[0]],
                     [0, 1, 0, v[1]],
                     [0, 0, 1, v[2]],
                     [0, 0, 0, 1]])


def m_proj_o(l, r, b, t, n, f):
    return np.array([[2 / (r - l), 0, 0, -(r + l) / (r - l)],
                     [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, 2 / (f - n), -(f + n) / (f - n)],
                     [0, 0, 0, 1]])


def m_view_port(x=0, y=0, w=1024, h=1024):
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


def back_face_culling(p, v0, v1, v2):
    v1 = np.array([v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]])
    v2 = np.array([v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]])
    v0 = np.array([v0[0] - p[0], v0[1] - p[1], v0[2] - p[2]])
    nn = norm(np.cross(v1, v2))
    s = np.inner(np.array([0, 0, -1]), nn)
    # s = np.inner(v0, nn)
    return s


def get_barycentric_coordinates(p, v0, v1, v2):
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


class LocalToWorld:
    def __init__(self, x_rotation_degrees: float, y_rotation_degrees: float, z_rotation_degrees: float,
                 transposition_vec: list, scaling_vec: list):
        self.x_rotation_degrees = x_rotation_degrees
        self.y_rotation_degrees = y_rotation_degrees
        self.z_rotation_degrees = z_rotation_degrees
        self.transposition_vec = transposition_vec
        self.scaling_vec = scaling_vec

    def get_matrix(self):
        return R(self.z_rotation_degrees, 'z').dot(R(self.y_rotation_degrees, 'y').dot(
            R(self.x_rotation_degrees, 'x').dot(T(self.transposition_vec).dot(S(self.scaling_vec)))))


class WorldToCamera:
    def __init__(self, camera_location_vec: list, camera_direction_vec: list):
        self.camera_location_vec = camera_location_vec
        self.camera_direction_vec = camera_direction_vec

    def get_matrix(self):
        cd = norm(np.array(self.camera_location_vec) - np.array(self.camera_direction_vec))
        up = np.array([0, 1, 0])
        cr = norm(np.cross(up, cd))
        cu = np.cross(cd, cr)
        L = np.array([[cr[0], cr[1], cr[2], 0],
                      [cu[0], cu[1], cu[2], 0],
                      [cd[0], cd[1], cd[2], 0],
                      [0, 0, 0, 1]])

        R = np.array([[1, 0, 0, -self.camera_location_vec[0]],
                      [0, 1, 0, -self.camera_location_vec[1]],
                      [0, 0, 1, -self.camera_location_vec[2]],
                      [0, 0, 0, 1]])

        return L.dot(R)


class Model:
    def __init__(self, obj_path: str, texture_path: str):
        self.obj_path = obj_path
        self.texture_path = texture_path

        with open(obj_path) as file:
            obj = file.readlines()

        self.texture_img = img.imread(texture_path)
        self.vertices = []
        self.new_vertices = []
        self.texture_v = []
        self.faces = []

        for line in obj:
            temp = line[:-1].split()
            if len(temp) == 0:
                continue
            elif temp[0] == 'v':
                self.vertices.append(np.array([float(temp[1]), float(temp[2]), float(temp[3]), 1]))
            elif temp[0] == 'vt':
                self.texture_v.append([float(temp[1]), float(temp[2])])
            elif temp[0] == 'f':
                cur1 = temp[1].split('/')
                cur2 = temp[2].split('/')
                cur3 = temp[3].split('/')
                self.faces.append([[int(cur1[0]) - 1, int(cur1[1]) - 1],
                                   [int(cur2[0]) - 1, int(cur2[1]) - 1],
                                   [int(cur3[0]) - 1, int(cur3[1]) - 1]])

    def create_camera_coordinates(self, l2w: LocalToWorld, w2c: WorldToCamera):
        m = w2c.get_matrix().dot(l2w.get_matrix())
        for i in range(len(self.vertices)):
            self.new_vertices.append(m.dot(self.vertices[i]))
        return self.new_vertices


def get_wire_image(view_height: int, view_width: int, image_height: int, image_width: int,
                   location_image_x: int, location_image_y: int, model: Model):
    image = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    image[:] = np.array([150, 150, 150])

    for j in range(location_image_y, location_image_y + image_height):
        for i in range(location_image_x, location_image_x + image_width):
            image[view_height - j - 1][i] = np.array([0, 0, 0], dtype=np.uint8)

    vertices = model.new_vertices
    faces = model.faces

    ll = min(vertices, key=lambda x: x[0])
    rr = max(vertices, key=lambda x: x[0])
    bb = min(vertices, key=lambda x: x[1])
    tt = max(vertices, key=lambda x: x[1])
    nn = min(vertices, key=lambda x: x[2])
    ff = max(vertices, key=lambda x: x[2])

    l = ll[0] - (rr[0] - ll[0]) / 20
    r = rr[0] + (rr[0] - ll[0]) / 20
    b = bb[1] - (tt[1] - bb[1]) / 20
    t = tt[1] + (tt[1] - bb[1]) / 20
    n = nn[2] - (ff[2] - nn[2]) / 20
    f = ff[2] + (ff[2] - nn[2]) / 20

    m = m_view_port(location_image_x, location_image_y, image_width, image_height).dot(m_proj_o(l, r, b, t, n, f))

    for face in faces:
        v1 = vertices[face[0][0]]
        v2 = vertices[face[1][0]]
        v3 = vertices[face[2][0]]

        v1 = m.dot(v1)
        v2 = m.dot(v2)
        v3 = m.dot(v3)

        draw_line(round(v1[0]), round(v1[1]), round(v2[0]), round(v2[1]), image, np.array([255, 255, 255]),
                  location_image_x, location_image_y, view_height)
        draw_line(round(v1[0]), round(v1[1]), round(v3[0]), round(v3[1]), image, np.array([255, 255, 255]),
                  location_image_x, location_image_y, view_height)
        draw_line(round(v2[0]), round(v2[1]), round(v3[0]), round(v3[1]), image, np.array([255, 255, 255]),
                  location_image_x, location_image_y, view_height)

    return image


def get_face_image(view_height: int, view_width: int, image_height: int, image_width: int,
                   location_image_x: int, location_image_y: int, model: Model):
    image = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    z_buffer = np.ones((view_height, view_width), dtype=np.float)
    image[:] = np.array([150, 150, 150])

    for j in range(location_image_y, location_image_y + image_height):
        for i in range(location_image_x, location_image_x + image_width):
            image[view_height - j - 1][i] = np.array([0, 0, 0], dtype=np.uint8)

    vertices = model.new_vertices
    faces = model.faces

    ll = min(vertices, key=lambda x: x[0])
    rr = max(vertices, key=lambda x: x[0])
    bb = min(vertices, key=lambda x: x[1])
    tt = max(vertices, key=lambda x: x[1])
    nn = min(vertices, key=lambda x: x[2])
    ff = max(vertices, key=lambda x: x[2])

    l = ll[0] - (rr[0] - ll[0]) / 20
    r = rr[0] + (rr[0] - ll[0]) / 20
    b = bb[1] - (tt[1] - bb[1]) / 20
    t = tt[1] + (tt[1] - bb[1]) / 20
    n = nn[2] - (ff[2] - nn[2]) / 20
    f = ff[2] + (ff[2] - nn[2]) / 20

    m = m_view_port(location_image_x, location_image_y, image_width, image_height).dot(m_proj_o(l, r, b, t, n, f))

    for face in faces:
        v1 = vertices[face[0][0]]
        v2 = vertices[face[1][0]]
        v3 = vertices[face[2][0]]

        color = back_face_culling([0, 0, 0], v1, v2, v3)

        if color >= 0:
            continue

        color = np.abs(color)

        color = np.array([color * 255, color * 255, color * 255])

        v1 = m.dot(v1)
        v2 = m.dot(v2)
        v3 = m.dot(v3)

        x_min = int(np.floor(min(v1, v2, v3, key=lambda x: x[0])[0]))
        x_max = int(np.ceil(max(v1, v2, v3, key=lambda x: x[0])[0]))
        y_min = int(np.floor(min(v1, v2, v3, key=lambda x: x[1])[1]))
        y_max = int(np.ceil(max(v1, v2, v3, key=lambda x: x[1])[1]))

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                a, b, c = get_barycentric_coordinates([i, j], v1, v2, v3)
                if a >= 0 and b >= 0 and c >= 0:
                    z = a * v1[2] + b * v2[2] + c * v3[2]
                    if -z < z_buffer[view_height - j, i]:
                        z_buffer[view_height - j, i] = -z
                        image[view_height - j][i] = color

    return image


def get_texture_image(view_height: int, view_width: int, image_height: int, image_width: int,
                      location_image_x: int, location_image_y: int, model: Model):
    image = np.zeros((view_height, view_width, 3), dtype=np.uint8)
    z_buffer = np.ones((view_height, view_width), dtype=np.float)
    image[:] = np.array([150, 150, 150])

    for j in range(location_image_y, location_image_y + image_height):
        for i in range(location_image_x, location_image_x + image_width):
            image[view_height - j - 1][i] = np.array([0, 0, 0], dtype=np.uint8)

    vertices = model.new_vertices
    texture_img = model.texture_img
    texture_v = model.texture_v
    faces = model.faces

    ll = min(vertices, key=lambda x: x[0])
    rr = max(vertices, key=lambda x: x[0])
    bb = min(vertices, key=lambda x: x[1])
    tt = max(vertices, key=lambda x: x[1])
    nn = min(vertices, key=lambda x: x[2])
    ff = max(vertices, key=lambda x: x[2])

    l = ll[0] - (rr[0] - ll[0]) / 20
    r = rr[0] + (rr[0] - ll[0]) / 20
    b = bb[1] - (tt[1] - bb[1]) / 20
    t = tt[1] + (tt[1] - bb[1]) / 20
    n = nn[2] - (ff[2] - nn[2]) / 20
    f = ff[2] + (ff[2] - nn[2]) / 20

    m = m_view_port(location_image_x, location_image_y, image_width, image_height).dot(m_proj_o(l, r, b, t, n, f))

    for face in faces:
        v1 = vertices[face[0][0]]
        v2 = vertices[face[1][0]]
        v3 = vertices[face[2][0]]

        if back_face_culling([0, 0, 0], v1, v2, v3) >= 0:
            continue

        v1 = m.dot(v1)
        v2 = m.dot(v2)
        v3 = m.dot(v3)

        x_min = int(np.floor(min(v1, v2, v3, key=lambda x: x[0])[0]))
        x_max = int(np.ceil(max(v1, v2, v3, key=lambda x: x[0])[0]))
        y_min = int(np.floor(min(v1, v2, v3, key=lambda x: x[1])[1]))
        y_max = int(np.ceil(max(v1, v2, v3, key=lambda x: x[1])[1]))

        for i in range(x_min, x_max + 1):
            for j in range(y_min, y_max + 1):
                a, b, c = get_barycentric_coordinates([i, j], v1, v2, v3)
                if a >= 0 and b >= 0 and c >= 0:
                    z = a * v1[2] + b * v2[2] + c * v3[2]
                    if -z < z_buffer[view_height - j, i]:
                        z_buffer[view_height - j, i] = -z
                        u = a * texture_v[face[0][1]][0] + b * texture_v[face[1][1]][0] + c * texture_v[face[2][1]][0]
                        v = a * texture_v[face[0][1]][1] + b * texture_v[face[1][1]][1] + c * texture_v[face[2][1]][1]
                        image[view_height - j][i] = texture_img[len(texture_img) - round(v * len(texture_img))][
                            round(u * len(texture_img[0]))]
    return image


def get_texture_image_with_with_light(view_height: int, view_width: int, image_height: int, image_width: int,
                                      location_image_x: int, location_image_y: int, model: Model):
    pass
