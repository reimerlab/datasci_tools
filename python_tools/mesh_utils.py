import copy
import trimesh
import numpy as np
import numpy_utils as nu


def mesh_center_vertex_average(mesh_list):
    if not nu.is_array_like(mesh_list):
        mesh_list = [mesh_list]
    mesh_list_centers = [np.array(np.mean(k.vertices,axis=0)).astype("float")
                           for k in mesh_list]
    if len(mesh_list) == 1:
        return mesh_list_centers[0]
    else:
        return mesh_list_centers

def translate_mesh(
    mesh,
    translation=None,
    new_center = None,
    inplace = False):
    
    if not inplace:
        mesh = copy.deepcopy(mesh)
    
    if translation is None:
        translation = new_center - mhu.mesh_center_vertex_average(mesh)
        
    mesh.vertices += translation
    return mesh

def scatter_mesh_with_radius(array,radius):
    """
    Purpose: to generate a mesh of spheres at certain coordinates with certain size

    """
    if not nu.is_array_like(radius):
        radius = [radius]*len(array)
    
    total_mesh = mhu.combine_meshes([mhu.sphere_mesh(k,r)
                    for k,r in zip(array,radius)])
    return total_mesh

def combine_meshes(mesh_pieces,merge_vertices=True):
    leftover_mesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))
#     for m in mesh_pieces:
#         leftover_mesh += m

    leftover_mesh = trimesh.util.concatenate( mesh_pieces +  [leftover_mesh])
        
    if merge_vertices:
        leftover_mesh.merge_vertices()
    
    return leftover_mesh

def combine_meshes(mesh_pieces,merge_vertices=True):
    leftover_mesh = trimesh.Trimesh(vertices=np.array([]),faces=np.array([]))
#     for m in mesh_pieces:
#         leftover_mesh += m

    leftover_mesh = trimesh.util.concatenate( mesh_pieces +  [leftover_mesh])
        
    if merge_vertices:
        leftover_mesh.merge_vertices()
    
    return leftover_mesh

def sphere_mesh(center,radius=100):
    sph = trimesh.creation.icosphere(subdivisions = 1,radius=radius)
    return mhu.center_mesh(sph,center)

def center_mesh(mesh,new_center):
    new_mesh = mesh
    added_offset =  np.array(new_center) - new_mesh.center_mass
    new_mesh.vertices = new_mesh.vertices + added_offset
    return new_mesh


import trimesh
import ipyvolume_utils as ipvu
from scipy.spatial import Delaunay
def mesh_from_delauny_3d(
    tri,
    plot = False,):
    
    if "array" in str(type(tri)):
        tri = Delaunay(tri)
    
    
    vertices = tri.points
    faces_orig = tri.simplices
    faces = np.vstack([
        faces_orig[:,[0,1,2]],
        faces_orig[:,[0,1,3]],
        faces_orig[:,[1,2,3]],
        faces_orig[:,[2,0,3]]]
    )
    
    
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces)

    if plot:
        ipvu.plot_mesh(mesh)
    return mesh

import mesh_utils as mhu