import numpy as np


def reformat_obj(input_file, output_file):
    f_in = open(input_file, "r")
    f_out = open(output_file, 'w')
    for line in f_in:
        line_list = line.split()
        if line_list[0] == 'f':
            f_0 = line_list[1].split('/')
            f_1 = line_list[2].split('/')
            f_2 = line_list[3].split('/')
            out_line = 'f ' + f_0[0] + ' ' + f_1[0] + ' ' + f_2[0] + '\n'
            f_out.write(out_line)
        else:
            f_out.write(line)
    return


def read_obj(f):
    """Read a file in the Wavefront OBJ file format.

    Argument is an open file.
    Returns a tuple of NumPy arrays: (indices, positions, normals, uvs).
    """

    # position, normal, uv, and face data in the order they appear in the file
    f_posns = []
    f_normals = []
    f_uvs = []
    f_faces = []

    # set of unique index combinations that appear in the file
    verts = set()

    # read file
    for words in (line.split() for line in f.readlines()):
        if words[0] == 'v':
            f_posns.append([float(s) for s in words[1:]])
        elif words[0] == 'vn':
            f_normals.append([float(s) for s in words[1:]])
        elif words[0] == 'vt':
            f_uvs.append([float(s) for s in words[1:]])
        elif words[0] == 'f':
            f_faces.append(words[1:])
            for w in words[1:]:
                verts.add(w)

    # there is one vertex for each unique index combo; number them
    vertmap = dict((s, i) for (i, s) in enumerate(sorted(verts)))

    # collate the vertex data for each vertex
    posns = [None] * len(vertmap)
    normals = [None] * len(vertmap)
    uvs = [None] * len(vertmap)
    for k, v in vertmap.items():
        w = k.split('/')
        posns[v] = f_posns[int(w[0]) - 1]
        # if len(w) > 1 and w[1]:
        #     uvs[v] = f_uvs[int(w[1]) - 1]
        if len(w) > 2 and w[2]:
            normals[v] = f_normals[int(w[2]) - 1]

    # set up faces using our ordering
    inds = [[vertmap[k] for k in f] for f in f_faces]

    # convert all to NumPy arrays with the right datatypes
    return (
        np.array(inds, dtype=np.int32),
        np.array(posns, dtype=np.float32),
        np.array(normals, dtype=np.float32),
        np.array(uvs, dtype=np.float32)
    )


def read_hs_obj_triangles(f):
    """Read a file in the Wavefront OBJ file format and convert to separate triangles.

    Argument is an open file.
    Returns an array of shape (n, 3, 3) that has the 3D vertex positions of n triangles.
    """

    (i, p, n, t) = read_obj(f)
    return p[i, :]

def read_hs_obj_triangles(f):
    """Read a file in the Wavefront OBJ file format and convert to separate triangles.

    Argument is an open file.
    Returns an array of shape (n, 3, 3) that has the 3D vertex positions of n triangles.
    """

    (i, p, n, t) = read_obj(f)
    return p[i, :]


def read_hs_obj_normals(f):
    """Read a file in the Wavefront OBJ file format and convert to separate triangles.

    Argument is an open file.
    Returns an array of shape (n, 3, 3) that has the 3D vertex positions of n triangles.
    """

    (i, p, n, t) = read_obj(f)
    return n[i, :]