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

def sphere_mesh(center,radius=100):
    sph = trimesh.creation.icosphere(subdivisions = 1,radius=radius)
    return mhu.center_mesh(sph,center)

def center_mesh(mesh,new_center):
    new_mesh = mesh
    added_offset =  np.array(new_center) - new_mesh.center_mass
    new_mesh.vertices = new_mesh.vertices + added_offset
    return new_mesh

def center_mesh_at_origin(mesh,center=None):
    new_mesh = mesh.copy()
    if center is None:
        center = new_mesh.center_mass
    
    added_offset = - center
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

def sample_surface(
    mesh,
    count,
    even_sampling=False,
    radius = None,
    plot=False):
    
    """
    Purpose: To sample the surface of a mesh
    
    import mesh_utils as mhu
    mhu.sample_surface(mesh,10000,even_sampling=False,plot=True)
    """
    
    if even_sampling:
        return_value,_ = trimesh.sample.sample_surface_even(mesh, count=count, radius=radius)
    else:
        return_value,_ = trimesh.sample.sample_surface(mesh,count=count)
        
    if plot:
        ipvu.plot_mesh_with_scatter(
            mesh,
            scatter = return_value,
        )
        
    return return_value
# -------------- 11/21 More bounding box functions ----- #
def bounding_box(mesh,oriented=False):
    """
    Returns the mesh of the bounding box
    
    Input: Can take in the corners of a bounding box as well
    """
    if nu.is_array_like(mesh):
        mesh = np.array(mesh).reshape(-1,3)
        if len(mesh) != 2:
            raise Exception("did not recieve bounding box corners")
        mesh = trimesh.Trimesh(vertices = np.vstack([mesh,mesh.mean(axis=0).reshape(-1,3)]).reshape(-1,3),
                              faces=np.array([[0,1,2]]))
        mesh = mesh.bounding_box
    
    if oriented:
        return mesh.bounding_box_oriented
    else:
        return mesh.bounding_box

def bounding_box_center(mesh,oriented=False):
    """
    Computed the center of the bounding box
    
    Ex:
    ex_mesh = neuron_obj_with_web[axon_limb_name][9].mesh
    nviz.plot_objects(ex_mesh,
                      scatters=[tu.bounding_box_center(ex_mesh)],
                      scatter_size=1)
    """
    bb_corners = bounding_box_corners(mesh,oriented = oriented)
    return np.mean(bb_corners,axis=0)
    
def bounding_box_corners(mesh,bbox_multiply_ratio=1,
                        oriented=False):
    #bbox_verts = mesh.bounding_box.vertices
    bbox_verts = bounding_box(mesh,oriented=oriented).vertices
    bb_corners = np.array([np.min(bbox_verts,axis=0),np.max(bbox_verts,axis=0)]).reshape(2,3)
    if bbox_multiply_ratio == 1:
        return bb_corners
    
    bbox_center = np.mean(bb_corners,axis=0)
    bbox_distance = np.max(bb_corners,axis=0)-bbox_center
    new_corners = np.array([bbox_center - bbox_multiply_ratio*bbox_distance,
                            bbox_center + bbox_multiply_ratio*bbox_distance
                           ]).reshape(-1,3)
    return new_corners

def bounding_box_diagonal(mesh):
    bbox_corners = bounding_box_corners(mesh)
    return np.linalg.norm(bbox_corners[1]-bbox_corners[0])
    
import mesh_utils as mhu