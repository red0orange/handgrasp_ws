import os, json
import trimesh
import mesh2sdf
import numpy as np
from scipy.spatial.transform import Rotation as R

def create_voxel_grid(mesh_path: str, n=32):
    """
    Uses the the mesh2sdf library to convert the mesh into a (n x n x n) voxel grid
    The values within the grid are sdf (signed distance function) values
    Input - 
        1. mesh_path --> Path to mesh file (.obj file)
        2. n --> size of voxel grid
    Output - 
        1. mesh --> mesh object after loading, normalizing and fixing mesh
        2. sdf --> (n x n x n) numpy array 
    """

    # Load mesh file
    mesh = trimesh.load(mesh_path, force='mesh')

    mesh_scale = 0.8
    size = n
    level = 2 / size

    # normalize mesh
    vertices = mesh.vertices
    bbmin = vertices.min(0)
    bbmax = vertices.max(0)
    center = (bbmin + bbmax) * 0.5
    scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
    vertices = (vertices - center) * scale

    sdf, mesh = mesh2sdf.compute(
        vertices, mesh.faces, size, fix=True, level=level, return_mesh=True)

    # output
    mesh.vertices = mesh.vertices / scale + center

    return mesh, sdf

def decompose_matrix(matrix):
    """
    Converts a transformation matrix into a translation vector and quaternion
    """

    # Extract translation vector
    translation = matrix[:3, 3]

    # Extract the 3x3 rotation matrix
    rotation_matrix = matrix[:3, :3]

    # Convert rotation matrix to quaternion
    quaternion = R.from_matrix(rotation_matrix).as_quat()

    return translation, quaternion

def compose_matrix(position, quaternion):
    """
    Converts a translation vector and quaternion into a transformation matrix
    """

    # Create a 3x3 rotation matrix from the quaternion
    rotation_matrix = R.from_quat(quaternion).as_matrix()

    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    
    # Set the rotation part of the matrix
    transformation_matrix[:3, :3] = rotation_matrix
    
    # Set the translation part of the matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix