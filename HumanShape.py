import numpy as np

from utils import *
from ray import *
from cli import render
from Read_HS_obj import reformat_obj, read_hs_obj_normals,read_hs_obj_triangles

tan = Material(vec([0.7, 0.7, 0.4]), 0.6)
gray = Material(vec([0.2, 0.2, 0.2]))

# Read the triangle mesh for a 2x2x2 cube, and scale it down to 1x1x1 to fit the scene.
new_file_name = "temp_humanshape.obj"
# reformat_obj("humanshape.obj", new_file_name)

vs_list = 0.001 * read_hs_obj_triangles(open("humanshape.obj"))
vs_normal_list = read_hs_obj_normals(open("humanshape.obj"))

# vs_list = 0.001 * read_obj_triangles(open(new_file_name))
# vs_list = 0.001 * read_obj_triangles(open('hs-test3000.obj'))

tranformation_matrix = np.zeros((4,4))
x_dir = np.array([0,0,-1])
y_dir = np.array([-1,0,0])
z_dir = np.array([0,1,0])
model_pos = np.array([-1.0,-1.3,0])
for i in range(3):
    tranformation_matrix[i,0] = x_dir[i]
    tranformation_matrix[i, 1] = y_dir[i]
    tranformation_matrix[i, 2] = z_dir[i]
    tranformation_matrix[i, 3] = model_pos[i]

deg_z = 75
theta_z = deg_z*np.pi/180
t_z = np.zeros((4,4))
t_z[0,0] = np.cos(theta_z)
t_z[2,0] = -np.sin(theta_z)
t_z[0,2] = np.sin(theta_z)
t_z[2,2] = np.cos(theta_z)
t_z[1,1] = 1
t_z[3,3] = 1


new_vs_list = []
for tris in vs_list:
    p_0 = tris[0]
    p_0_h = np.array([[p_0[0]],[p_0[1]], [p_0[2]], [1]])
    new_p0_h = t_z.dot(tranformation_matrix.dot(p_0_h))

    p_1 = tris[1]
    p_1_h = np.array([[p_1[0]], [p_1[1]], [p_1[2]], [1]])
    new_p1_h = t_z.dot(tranformation_matrix.dot(p_1_h))

    p_2 = tris[2]
    p_2_h = np.array([[p_2[0]], [p_2[1]], [p_2[2]], [1]])
    new_p2_h = t_z.dot(tranformation_matrix.dot(p_2_h))
    # new_vs_list.append(np.array([[new_p0_h[0,0], new_p0_h[1,0], new_p0_h[2,0]],
    #                              [new_p1_h[0,0], new_p1_h[1,0], new_p1_h[2,0]],
    #                              [new_p2_h[0,0], new_p2_h[1,0], new_p2_h[2,0]]]))
    new_vs_list.append(tris)

new_vn_list = []
for tris in vs_normal_list:
    p_0 = tris[0]
    p_0_h = np.array([[p_0[0]],[p_0[1]], [p_0[2]], [1]])
    new_p0_h = t_z.dot(tranformation_matrix.dot(p_0_h))

    p_1 = tris[1]
    p_1_h = np.array([[p_1[0]], [p_1[1]], [p_1[2]], [1]])
    new_p1_h = t_z.dot(tranformation_matrix.dot(p_1_h))

    p_2 = tris[2]
    p_2_h = np.array([[p_2[0]], [p_2[1]], [p_2[2]], [1]])
    new_p2_h = t_z.dot(tranformation_matrix.dot(p_2_h))
    # new_vn_list.append(np.array([[new_p0_h[0,0], new_p0_h[1,0], new_p0_h[2,0]],
    #                              [new_p1_h[0,0], new_p1_h[1,0], new_p1_h[2,0]],
    #                              [new_p2_h[0,0], new_p2_h[1,0], new_p2_h[2,0]]]))
    new_vn_list.append(tris)


tri_list = []

for i in range(len(new_vs_list)):
    tri_list.append(Triangle(new_vs_list[i],tan,new_vn_list[i],True))



scene = Scene(
    [
    # Make a big sphere for the floor
    Sphere(vec([0,-40,0]), 39.5, gray),
] +tri_list
#               [
#     # Make triangle objects from the vertex coordinates
#     Triangle(vs, tan) for vs in new_vs_list
# ]
)

lights = [
    PointLight(vec([12,10,7]), vec([300,300,300])),
    AmbientLight(0.1),
]

camera = Camera(vec([3,1.2,5]), target=vec([0,-0.4,0]), vfov=25, aspect=16/9)

render(camera, scene, lights)



