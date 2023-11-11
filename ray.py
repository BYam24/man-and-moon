import numpy as np

from utils import *

"""
Core implementation of the ray tracer.  This module contains the classes (Sphere, Mesh, etc.)
that define the contents of scenes, as well as classes (Ray, Hit) and functions (shade) used in 
the rendering algorithm, and the main entry point `render_image`.

In the documentation of these classes, we indicate the expected types of arguments with a
colon, and use the convention that just writing a tuple means that the expected type is a
NumPy array of that shape.  Implementations can assume these types are preconditions that
are met, and if they fail for other type inputs it's an error of the caller.  (This might 
not be the best way to handle such validation in industrial-strength code but we are adopting
this rule to keep things simple and efficient.)
"""

MAX_DEPTH = 4
MAX_LEAF_ELE = 7
small_delta_for_shadow = 0.0001


class Ray:

    def __init__(self, origin, direction, start=0., end=np.inf):
        """Create a ray with the given origin and direction.

        Parameters:
          origin : (3,) -- the start point of the ray, a 3D point
          direction : (3,) -- the direction of the ray, a 3D vector (not necessarily normalized)
          start, end : float -- the minimum and maximum t values for intersections
        """
        # Convert these vectors to double to help ensure intersection
        # computations will be done in double precision
        self.origin = np.array(origin, np.float64)
        self.direction = np.array(direction, np.float64)
        self.start = start
        self.end = end


class Material:

    def __init__(self, k_d, k_s=0., p=20., k_m=0., k_a=None):
        """Create a new material with the given parameters.

        Parameters:
          k_d : (3,) -- the diffuse coefficient
          k_s : (3,) or float -- the specular coefficient
          p : float -- the specular exponent
          k_m : (3,) or float -- the mirror reflection coefficient
          k_a : (3,) -- the ambient coefficient (defaults to match diffuse color)
        """
        self.k_d = k_d
        self.k_s = k_s
        self.p = p
        self.k_m = k_m
        self.k_a = k_a if k_a is not None else k_d


class Hit:

    def __init__(self, t, point=None, normal=None, material=None):
        """Create a Hit with the given data.

        Parameters:
          t : float -- the t value of the intersection along the ray
          point : (3,) -- the 3D point where the intersection happens
          normal : (3,) -- the 3D outward-facing unit normal to the surface at the hit point
          material : (Material) -- the material of the surface
        """
        self.t = t
        self.point = point
        self.normal = normal
        self.material = material


# Value to represent absence of an intersection
no_hit = Hit(np.inf)


class Sphere:

    def __init__(self, center, radius, material):
        """Create a sphere with the given center and radius.

        Parameters:
          center : (3,) -- a 3D point specifying the sphere's center
          radius : float -- a Python float specifying the sphere's radius
          material : Material -- the material of the surface
        """
        self.center = center
        self.radius = radius
        self.material = material

        self.x_max = center[0] + radius
        self.x_min = center[0] - radius
        self.y_max = center[1] + radius
        self.y_min = center[1] - radius
        self.z_max = center[2] + radius
        self.z_min = center[2] - radius

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        p = ray.origin - self.center
        v = ray.direction
        c = p.transpose().dot(p) - self.radius * self.radius
        b = 2 * p.transpose().dot(v)
        a = v.transpose().dot(v)
        delta = b * b - 4 * a * c
        if delta < 0:
            return no_hit
        else:
            t1 = (-b - np.sqrt(delta)) / (2 * a)
            t2 = (-b + np.sqrt(delta)) / (2 * a)
            if ray.start < t1 < ray.end:
                hit_t = t1
                intersect_h = p + t1 * v
                hit_intersect = intersect_h + self.center
                hit_norm = normalize_vec3(intersect_h)
                hit_material = self.material
                return Hit(hit_t, hit_intersect, hit_norm, hit_material)
            elif ray.start < t2 < ray.end:
                hit_t = t2
                intersect_h = p + t2 * v
                hit_intersect = intersect_h + self.center
                hit_norm = normalize_vec3(intersect_h)
                hit_material = self.material
                return Hit(hit_t, hit_intersect, hit_norm, hit_material)

        return no_hit


class Triangle:

    def __init__(self, vs, material):
        """Create a triangle from the given vertices.

        Parameters:
          vs (3,3) -- an arry of 3 3D points that are the vertices (CCW order)
          material : Material -- the material of the surface
        """
        self.vs = vs
        self.material = material

        self.x_max = np.max([vs[0, 0], vs[1, 0], vs[2, 0]])
        self.x_min = np.min([vs[0, 0], vs[1, 0], vs[2, 0]])
        self.y_max = np.max([vs[0, 1], vs[1, 1], vs[2, 1]])
        self.y_min = np.min([vs[0, 1], vs[1, 1], vs[2, 1]])
        self.z_max = np.max([vs[0, 2], vs[1, 2], vs[2, 2]])
        self.z_min = np.min([vs[0, 2], vs[1, 2], vs[2, 2]])

    def intersect(self, ray):
        """Computes the intersection between a ray and this triangle, if it exists.

        Parameters:
          ray : Ray -- the ray to intersect with the triangle
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        p = ray.origin
        d = ray.direction
        a = vec(self.vs[0])
        b = vec(self.vs[2])
        c = vec(self.vs[1])
        A = np.zeros((3, 3))
        RHS = np.zeros((3, 1))
        for i in range(3):
            A[i, 0] = d[i]
            A[i, 1] = a[i] - b[i]
            A[i, 2] = a[i] - c[i]
            RHS[i, 0] = a[i] - p[i]
        result = np.linalg.inv(A).dot(RHS)
        t = result[0, 0]
        beta = result[1, 0]
        gamma = result[2, 0]

        if beta > 0 and gamma > 0 and beta + gamma < 1:
            if ray.start < t < ray.end:
                intersection_pos = p + d * t
                hit_norm = normalize_vec3(np.cross(c - a, b - a))
                return Hit(t, intersection_pos, hit_norm, self.material)

        return no_hit


class Camera:

    def __init__(self, eye=vec([0, 0, 0]), target=vec([0, 0, -1]), up=vec([0, 1, 0]),
                 vfov=90.0, aspect=1.0):
        """Create a camera with given viewing parameters.

        Parameters:
          eye : (3,) -- the camera's location, aka viewpoint (a 3D point)
          target : (3,) -- where the camera is looking: a 3D point that appears centered in the view
          up : (3,) -- the camera's orientation: a 3D vector that appears straight up in the view
          vfov : float -- the full vertical field of view in degrees
          aspect : float -- the aspect ratio of the camera's view (ratio of width to height)
        """
        self.eye = eye
        self.aspect = aspect
        half_theta = (vfov / 2) / 180 * np.pi
        self.f = 1 / np.tan(
            half_theta)  # you should set this to the distance from your center of projection to the image plane
        self.M = np.zeros(
            (4, 4))  # set this to the matrix that transforms your camera's coordinate system to world coordinates

        # TODO A4 implement this constructor to store whatever you need for ray generation
        # y_dir = normalize_vec3(up)
        z_dir = normalize_vec3(-(target - eye))
        x_dir = normalize_vec3(np.cross(target - eye, up))
        y_dir = normalize_vec3(np.cross(z_dir, x_dir))
        for i in range(3):
            self.M[i, 0] = x_dir[i]
            self.M[i, 1] = y_dir[i]
            self.M[i, 2] = z_dir[i]
            self.M[i, 3] = eye[i]
        self.M[3, 3] = 1

    def generate_ray(self, img_point):
        """Compute the ray corresponding to a point in the image.

        Parameters:
          img_point : (2,) -- a 2D point in [0,1] x [0,1], where (0,0) is the upper left
                      corner of the image and (1,1) is the lower right.
                      (note: since we initially released this code with specs saying 0,0 was at the bottom left, we will
                       accept either convention for this assignment)
        Return:
          Ray -- The ray corresponding to that image location (not necessarily normalized)
        """
        # TODO A4 implement this function
        x_on_img_plane = (img_point[0] - 0.5) * 2 * self.aspect
        y_on_img_plane = -(img_point[1] - 0.5) * 2
        img_point_in_camera = np.array([[x_on_img_plane], [y_on_img_plane], [-self.f], [1]])
        img_point_in_world = self.M.dot(img_point_in_camera)
        img_point_in_world_vec3 = vec([img_point_in_world[0, 0], img_point_in_world[1, 0], img_point_in_world[2, 0]])
        return Ray(self.eye, normalize_vec3(img_point_in_world_vec3 - self.eye))


class PointLight:

    def __init__(self, position, intensity):
        """Create a point light at given position and with given intensity

        Parameters:
          position : (3,) -- 3D point giving the light source location in scene
          intensity : (3,) or float -- RGB or scalar intensity of the source
        """
        self.position = position
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function
        blocked = False
        color = vec([0, 0, 0])
        shadow_ray_dir = normalize_vec3(self.position - hit.point)
        shadow_ray = Ray(hit.point, shadow_ray_dir, start=small_delta_for_shadow)
        scene_shadow_intersect = scene.intersect(shadow_ray)
        if scene_shadow_intersect != no_hit:
            blocked = True

        if not blocked:
            n = hit.normal
            hit_to_source = self.position - hit.point
            l = normalize_vec3(hit_to_source)
            r_sq = hit_to_source[0] * hit_to_source[0] + hit_to_source[1] * hit_to_source[1] + hit_to_source[2] * \
                   hit_to_source[2]
            n_dot_l = n.dot(l)
            irradiance = (np.max([0, n_dot_l]) / r_sq) * self.intensity
            v = -ray.direction
            h = normalize_vec3(v + l)
            n_dot_h = n.dot(h)
            n_dot_h_pow = np.power(n_dot_h, hit.material.p)
            color = np.multiply(hit.material.k_d + hit.material.k_s * n_dot_h_pow, irradiance)

        return color


class AmbientLight:

    def __init__(self, intensity):
        """Create an ambient light of given intensity

        Parameters:
          intensity (3,) or float: the intensity of the ambient light
        """
        self.intensity = intensity

    def illuminate(self, ray, hit, scene):
        """Compute the shading at a surface point due to this light.

        Parameters:
          ray : Ray -- the ray that hit the surface
          hit : Hit -- the hit data
          scene : Scene -- the scene, for shadow rays
        Return:
          (3,) -- the light reflected from the surface
        """
        # TODO A4 implement this function

        return np.multiply(hit.material.k_a, self.intensity)


class Box:
    def __init__(self, ):
        self.x_min = 0
        self.x_max = 0
        self.y_min = 0
        self.y_max = 0
        self.z_min = 0
        self.z_max = 0
        self.temp = []

    def set_ax(self, axis, min_v, max_v):
        if axis == 0:
            self.x_min = min_v
            self.x_max = max_v
        elif axis == 1:
            self.y_min = min_v
            self.y_max = max_v
        elif axis == 2:
            self.z_min = min_v
            self.z_max = max_v

    def generate_ax_table(self):
        return np.array([[self.x_min, self.x_max], [self.y_min, self.y_max], [self.z_min, self.z_max]])

    def set_by_ax_table(self, ax_table):
        for i in range(3):
            self.set_ax(i, ax_table[i, 0], ax_table[i, 1])
        return

    def point_in_box(self, point):
        if point[0]< self.x_min-small_delta_for_shadow or point[0] > self.x_max+small_delta_for_shadow:
            return False
        if point[1] < self.y_min-small_delta_for_shadow or point[1] > self.y_max+small_delta_for_shadow:
            return False
        if point[2]< self.z_min-small_delta_for_shadow or point[2] > self.z_max+small_delta_for_shadow:
            return False
        return True

    def has_intersect(self, ray):
        self.temp = []
        ax_table = self.generate_ax_table()
        for i in range(3):
            if ray.direction[i] != 0:
                t1 = (ax_table[i, 0] - ray.origin[i]) / ray.direction[i]

                if ray.start-0.01 < t1 < ray.end+0.01:
                    hit_point = ray.origin + ray.direction * t1
                    self.temp.append(hit_point)
                    if self.point_in_box(hit_point):
                        return True

                t2 = (ax_table[i, 1] - ray.origin[i]) / ray.direction[i]
                if ray.start-0.01 < t2 < ray.end+0.01:
                    hit_point = ray.origin + ray.direction * t2
                    self.temp.append(hit_point)
                    if self.point_in_box(hit_point):
                        return True

        return False


# def compare_surf_x(s1, s2):
#     return s1.x_max - s2.x_max
#
#
# def compare_surf_y(s1, s2):
#     return s1.y_max - s2.y_max
#
#
# def compare_surf_z(s1, s2):
#     return s1.z_max - s2.z_max




class Node:
    def __init__(self, sep_ax):
        self.box = Box()
        self.left_node = None
        self.right_node = None
        self.surfs = None
        self.is_leaf = True
        self.cur_sep_ax = sep_ax
        self.temp=[]

    def set_surfs(self, surface_list):
        if len(surface_list) <= MAX_LEAF_ELE:
            self.surfs = surface_list
        else:
            self.is_leaf = False
            if self.cur_sep_ax == 0:
                surface_list.sort(key=lambda x: x.x_max)
            elif self.cur_sep_ax == 1:
                surface_list.sort(key=lambda x: x.y_max)
            elif self.cur_sep_ax == 2:
                surface_list.sort(key=lambda x: x.z_max)

            mid = len(surface_list) // 2

            left_surface_list = surface_list[0:mid]
            right_surface_list = surface_list[mid:len(surface_list)]

            next_sep_ax = self.get_next_sep_ax()

            self.left_node = Node(next_sep_ax)
            x_min = (min(left_surface_list, key=lambda x: x.x_min)).x_min
            x_max = (max(left_surface_list, key=lambda x: x.x_max)).x_max
            y_min = (min(left_surface_list, key=lambda x: x.y_min)).y_min
            y_max = (max(left_surface_list, key=lambda x: x.y_max)).y_max
            z_min = (min(left_surface_list, key=lambda x: x.z_min)).z_min
            z_max = (max(left_surface_list, key=lambda x: x.z_max)).z_max
            left_node_table = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
            # left_node_table = self.box.generate_ax_table()
            # if self.cur_sep_ax == 0:
            #     left_node_table[0,1] = (max(left_surface_list, key=lambda x: x.x_max)).x_max
            # elif self.cur_sep_ax == 1:
            #     left_node_table[1, 1] = (max(left_surface_list, key=lambda x: x.y_max)).y_max
            # elif self.cur_sep_ax == 2:
            #     left_node_table[2, 1] = (max(left_surface_list, key=lambda x: x.z_max)).z_max
            self.left_node.box.set_by_ax_table(left_node_table)
            self.left_node.set_surfs(left_surface_list)

            self.right_node = Node(next_sep_ax)
            x_min = (min(right_surface_list, key=lambda x: x.x_min)).x_min
            x_max = (max(right_surface_list, key=lambda x: x.x_max)).x_max
            y_min = (min(right_surface_list, key=lambda x: x.y_min)).y_min
            y_max = (max(right_surface_list, key=lambda x: x.y_max)).y_max
            z_min = (min(right_surface_list, key=lambda x: x.z_min)).z_min
            z_max = (max(right_surface_list, key=lambda x: x.z_max)).z_max
            right_node_table = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
            # right_node_table = self.box.generate_ax_table()
            # if self.cur_sep_ax == 0:
            #     right_node_table[0,0] = (min(right_surface_list, key=lambda x: x.x_min)).x_min
            # elif self.cur_sep_ax == 1:
            #     right_node_table[1,0] = (min(right_surface_list, key=lambda x: x.y_min)).y_min
            # elif self.cur_sep_ax == 2:
            #     right_node_table[2,0] = (min(right_surface_list, key=lambda x: x.z_min)).z_min
            self.right_node.box.set_by_ax_table(right_node_table)
            self.right_node.set_surfs(right_surface_list)

    def get_next_sep_ax(self):
        return (self.cur_sep_ax + 1) % 3

    def intersection(self,ray):
        self.temp=[]
        if self.is_leaf:
            res = no_hit
            t_min = np.inf
            for surf in self.surfs:
                temp = surf.intersect(ray)
                if temp != no_hit:
                    if temp.t < t_min:
                        res = temp
                        t_min = temp.t

            return res
        else:
            if not self.box.has_intersect(ray):

                return no_hit
            left_hit = self.left_node.intersection(ray)
            right_hit = self.right_node.intersection(ray)
            if left_hit == no_hit and right_hit == no_hit:
                return no_hit
            elif left_hit == no_hit:
                return right_hit
            elif right_hit == no_hit:
                return left_hit
            else:
                if right_hit.t<left_hit.t:
                    return right_hit
                else:
                    return left_hit




class Scene:

    def __init__(self, surfs, bg_color=vec([0.2, 0.3, 0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color
        first_sep_ax = 1
        self.root_node = Node(first_sep_ax)
        x_min = (min(surfs, key=lambda x: x.x_min)).x_min
        x_max = (max(surfs, key=lambda x: x.x_max)).x_max
        y_min = (min(surfs, key=lambda x: x.y_min)).y_min
        y_max = (max(surfs, key=lambda x: x.y_max)).y_max
        z_min = (min(surfs, key=lambda x: x.z_min)).z_min
        z_max = (max(surfs, key=lambda x: x.z_max)).z_max
        ax_minmax_table = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        self.root_node.box.set_by_ax_table(ax_minmax_table)
        self.root_node.set_surfs(self.surfs)




    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        # res = no_hit
        # t_min = np.inf
        # for surf in self.surfs:
        #     temp = surf.intersect(ray)
        #     if temp != no_hit:
        #         if temp.t < t_min:
        #             res = temp
        #             t_min = temp.t
        #
        # return res
        return self.root_node.intersection(ray)


# MAX_DEPTH = 4


def shade(ray, hit, scene, lights, depth=0):
    """Compute shading for a ray-surface intersection.

    Parameters:
      ray : Ray -- the ray that hit the surface
      hit : Hit -- the hit data
      scene : Scene -- the scene
      lights : [PointLight or AmbientLight] -- the lights
      depth : int -- the recursion depth so far
    Return:
      (3,) -- the color seen along this ray
    When mirror reflection is being computed, recursion will only proceed to a depth
    of MAX_DEPTH, with zero contribution beyond that depth.
    """
    # TODO A4 implement this function
    color = vec([0, 0, 0])
    for light in lights:
        color += light.illuminate(ray, hit, scene)

    if depth <= MAX_DEPTH:
        # get mirror value
        v = - ray.direction
        n = hit.normal
        reflected_dir = 2 * (n.dot(v)) * n - v
        reflected_ray = Ray(hit.point, reflected_dir, start=small_delta_for_shadow)
        reflection_intersection = scene.intersect(reflected_ray)
        if reflection_intersection != no_hit:
            L_r = shade(reflected_ray, reflection_intersection, scene, lights, depth + 1)
            color += np.multiply(hit.material.k_m, L_r)

    return color


def texture_to_image_plane(w, h, x, y):
    x_res = (x - w / 2) / (w / 2)
    y_res = -(y - h / 2) / (h / 2)
    return [x_res, y_res]


def normalize_vec3(v):
    n = np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return v / n




def render_image(camera, scene, lights, nx, ny):
    """Render a ray traced image.

    Parameters:
      camera : Camera -- the camera defining the view
      scene : Scene -- the scene to be rendered
      lights : Lights -- the lights illuminating the scene
      nx, ny : int -- the dimensions of the rendered image
    Returns:
      (ny, nx, 3) float32 -- the RGB image
    """
    # TODO A4 implement this function

    output_image = np.zeros((ny, nx, 3), np.float32)
    test_ray = Ray(vec([0, 0, 0]), vec([0, -1, 0]))
    # print(scene.intersect(test_ray).point)
    for i in range(ny):
        if i%20 == 0:
            print("Arriving at: " + str(i) +" row")
        for j in range(nx):
            # [x_on_plane,y_on_plane] = texture_to_image_plane(nx,ny,j,i)
            ray = camera.generate_ray(vec([j / nx, i / ny]))
            intersection = scene.intersect(ray)  # this will return a Hit object
            if intersection != no_hit:
                material_at_pixel = intersection.material
                output_image[i, j] = shade(ray, intersection, scene, lights)
            else:
                output_image[i, j] = scene.bg_color

            # set the output pixel color if an intersection is found
            # ...
    output_image = np.clip(output_image, 0, 255)

    return output_image
