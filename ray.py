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

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and this sphere.

        Parameters:
          ray : Ray -- the ray to intersect with the sphere
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        p = ray.origin-self.center
        v = ray.direction
        c = p.transpose().dot(p)-self.radius*self.radius
        b = 2* p.transpose().dot(v)
        a = v.transpose().dot(v)
        delta = b*b-4*a*c
        if delta<0:
            return no_hit
        else:
            t1 = (-b-np.sqrt(delta))/(2*a)
            t2 = (-b + np.sqrt(delta)) / (2 * a)
            if ray.start < t1 < ray.end:
                hit_t = t1
                intersect_h = p+t1*v
                hit_intersect = intersect_h+self.center
                hit_norm = normalize_vec3(intersect_h)
                hit_material = self.material
                return Hit(hit_t,hit_intersect,hit_norm,hit_material)
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
        A=np.zeros((3,3))
        RHS = np.zeros((3,1))
        for i in range(3):
            A[i,0] = d[i]
            A[i,1] = a[i]-b[i]
            A[i,2] = a[i]-c[i]
            RHS[i,0] = a[i]-p[i]
        result = np.linalg.inv(A).dot(RHS)
        t = result[0,0]
        beta = result[1,0]
        gamma = result[2,0]

        if beta>0 and gamma>0 and beta+gamma<1:
            if ray.start< t < ray.end:
                intersection_pos = p+d*t
                hit_norm = normalize_vec3(np.cross(c-a, b-a))
                return Hit(t, intersection_pos, hit_norm, self.material)



        return no_hit


class Camera:

    def __init__(self, eye=vec([0,0,0]), target=vec([0,0,-1]), up=vec([0,1,0]), 
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
        half_theta = (vfov/2)/180*np.pi
        self.f = 1/np.tan(half_theta) # you should set this to the distance from your center of projection to the image plane
        self.M = np.zeros((4,4))  # set this to the matrix that transforms your camera's coordinate system to world coordinates

        # TODO A4 implement this constructor to store whatever you need for ray generation
        # y_dir = normalize_vec3(up)
        z_dir = normalize_vec3(-(target-eye))
        x_dir = normalize_vec3(np.cross(target-eye,up))
        y_dir = normalize_vec3(np.cross(z_dir,x_dir))
        for i in range(3):
            self.M[i, 0] = x_dir[i]
            self.M[i, 1] = y_dir[i]
            self.M[i, 2] = z_dir[i]
            self.M[i, 3] = eye[i]
        self.M[3,3] = 1

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
        x_on_img_plane = (img_point[0]-0.5)*2*self.aspect
        y_on_img_plane = -(img_point[1]-0.5)*2
        img_point_in_camera = np.array([[x_on_img_plane],[y_on_img_plane],[-self.f],[1]])
        img_point_in_world = self.M.dot(img_point_in_camera)
        img_point_in_world_vec3 = vec([img_point_in_world[0,0],img_point_in_world[1,0],img_point_in_world[2,0]])
        return Ray(self.eye, normalize_vec3(img_point_in_world_vec3-self.eye))


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
        color = vec([0,0,0])
        shadow_ray_dir = normalize_vec3(self.position-hit.point)
        shadow_ray = Ray(hit.point,shadow_ray_dir,start=small_delta_for_shadow)
        scene_shadow_intersect = scene.intersect(shadow_ray)
        if scene_shadow_intersect != no_hit:
            blocked = True


        if not blocked:
            n = hit.normal
            hit_to_source = self.position - hit.point
            l = normalize_vec3(hit_to_source)
            r_sq = hit_to_source[0]*hit_to_source[0]+hit_to_source[1]*hit_to_source[1]+hit_to_source[2]*hit_to_source[2]
            n_dot_l = n.dot(l)
            irradiance = (np.max([0,n_dot_l])/r_sq)*self.intensity
            v = -ray.direction
            h = normalize_vec3(v+l)
            n_dot_h = n.dot(h)
            n_dot_h_pow = np.power(n_dot_h,hit.material.p)
            color = np.multiply(hit.material.k_d + hit.material.k_s*n_dot_h_pow, irradiance)

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

        return np.multiply(hit.material.k_a,self.intensity)


class Scene:

    def __init__(self, surfs, bg_color=vec([0.2,0.3,0.5])):
        """Create a scene containing the given objects.

        Parameters:
          surfs : [Sphere, Triangle] -- list of the surfaces in the scene
          bg_color : (3,) -- RGB color that is seen where no objects appear
        """
        self.surfs = surfs
        self.bg_color = bg_color

    def intersect(self, ray):
        """Computes the first (smallest t) intersection between a ray and the scene.

        Parameters:
          ray : Ray -- the ray to intersect with the scene
        Return:
          Hit -- the hit data
        """
        # TODO A4 implement this function
        res = no_hit
        t_min = np.inf
        for surf in self.surfs:
            temp = surf.intersect(ray)
            if temp != no_hit:
                if temp.t<t_min:
                    res = temp
                    t_min = temp.t

        return res


MAX_DEPTH = 4

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
    color = vec([0,0,0])
    for light in lights:
        color+=light.illuminate(ray, hit, scene)

    if depth<=MAX_DEPTH:
        # get mirror value
        v= - ray.direction
        n = hit.normal
        reflected_dir = 2*(n.dot(v))*n-v
        reflected_ray = Ray(hit.point,reflected_dir,start=small_delta_for_shadow)
        reflection_intersection = scene.intersect(reflected_ray)
        if reflection_intersection != no_hit:
            L_r = shade(reflected_ray,reflection_intersection,scene,lights,depth+1)
            color+= np.multiply(hit.material.k_m, L_r)

    return color


def texture_to_image_plane(w,h,x,y):
    x_res = (x-w/2)/(w/2)
    y_res = -(y-h/2)/(h/2)
    return [x_res,y_res]

def normalize_vec3(v):
    n = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    return v/n

small_delta_for_shadow = 0.0001

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
    for i in range(ny):
        for j in range(nx):
            # [x_on_plane,y_on_plane] = texture_to_image_plane(nx,ny,j,i)
            ray = camera.generate_ray(vec([j/nx,i/ny]))
            intersection = scene.intersect(ray)  # this will return a Hit object
            if intersection!= no_hit:
                material_at_pixel = intersection.material
                output_image[i,j] = shade(ray,intersection, scene, lights)
            else:
                output_image[i, j] = scene.bg_color

            # set the output pixel color if an intersection is found
            # ...
    output_image = np.clip(output_image,0,255)

    return output_image
