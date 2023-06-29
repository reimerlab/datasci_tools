
from scipy.spatial import Delaunay
import copy
import math
import numpy
import numpy as np
import scipy.cluster
import scipy.linalg
import scipy.sparse
import scipy.sparse.csgraph
import scipy.sparse.linalg
import time
import trimesh
#import trimesh
#import numpy as np
#from python_tools import numpy_utils as nu


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
        translation = new_center - mesh_center_vertex_average(mesh)
        
    mesh.vertices += translation
    return mesh

def scatter_mesh_with_radius(array,radius):
    """
    Purpose: to generate a mesh of spheres at certain coordinates with certain size

    """
    if not nu.is_array_like(radius):
        radius = [radius]*len(array)
    
    total_mesh = combine_meshes([sphere_mesh(k,r)
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
    return center_mesh(sph,center)

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
    


#import trimesh
#from python_tools import ipyvolume_utils as ipvu
#from scipy.spatial import Delaunay
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
    
    from python_tools import mesh_utils as mhu
    sample_surface(mesh,10000,even_sampling=False,plot=True)
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

def bounding_box_side_lengths(
    mesh,
    oriented = False,
    sort_by_length = False,
    sort_descending = True,
    verbose = False,):
    
    bbox_corners = bounding_box_corners(mesh,oriented=oriented)
    side_lengths = [np.abs(bbox_corners[0][ax] - bbox_corners[1][ax])
                   for ax in [0,1,2]]
    
    
    
    if sort_by_length:
        side_lengths = np.sort(side_lengths)
        if sort_descending:
            side_lengths = np.flip(side_lengths)
            
    if verbose:
        print(f"Side lengths = {side_lengths}")
            
    return side_lengths

def bounding_box_side_lengths_sorted(
    mesh,
    oriented = False,
    sort_descending = True,
    verbose = False,
    ):
    return bounding_box_side_lengths(
    mesh,
    oriented = oriented,
    sort_by_length = True,
    sort_descending = sort_descending,
    verbose = verbose,)
    
    
    
    
def normal_vector_from_face_idx(mesh,face_idx):
    return mesh.face_normals[face_idx]


# ------------ 11/3 Implementing a mesh segmentation algorithm ----------
"""
Source Code: https://github.com/kugelrund/mesh_segmentation
Reference paper: https://www.cs.sfu.ca/~haoz/pubs/liu_zhang_pg04.pdf
"""
#import math
#import numpy
#import scipy.linalg
#import scipy.cluster
#import scipy.sparse
#import scipy.sparse.csgraph
#import scipy.sparse.linalg

delta_default = 0.03
eta_default = 0.15
def face_center(mesh,face):
    """
    Purpose: To compute the center of a face
    """
    return mesh.triangles_center[face]

def geodesic_distance(mesh,face1,face2,edge):
    edge_center = np.mean(mesh.vertices[edge],axis=0)
    return np.sum([np.linalg.norm(edge_center - face_center(mesh,k))
                  for k in [face1,face2]])

def angular_distance(
    mesh,
    face1,
    face2,
    eta = eta_default):
    v1 = meshu.normal_vector_from_face_idx(mesh,face1)
    v2 =  meshu.normal_vector_from_face_idx(mesh,face2)
    ang_distance = (1 - math.cos(nu.angle_between_vectors_simple(v1,v2)))
    
    if (v1@(face_center(mesh,face1)-face_center(mesh,face2))) < 0:
        ang_distance *= eta
        
    return ang_distance

def edge_to_faces_map(mesh):
    return {tuple(k):v for k,v in zip(
        mesh.face_adjacency_edges,
        mesh.face_adjacency
    )}


def create_distance_matrix(
    mesh,
    delta = delta_default,
    eta = eta_default):
    """Creates the matrix of the angular and geodesic distances
    between all adjacent faces. The i,j-th entry of the returned
    matrices contains the distance between the i-th and j-th face.
    """

    faces = mesh.faces
    l = len(faces)
    
    '''
    Old method
    adj_faces_map = {}
    # find adjacent faces by iterating edges
    for index, face in enumerate(faces):
        for edge in face.edge_keys:
            if edge in adj_faces_map:
                adj_faces_map[edge].append(index)
            else:
                adj_faces_map[edge] = [index]
    '''

    # map from edge-key to adjacent faces
    adj_faces_map = meshu.edge_to_faces_map(mesh)

    # helping vectors to create sparse matrix later on
    row_indices = []
    col_indices = []
    Gval = []  # values for matrix of angular distances
    Aval = []  # values for matrix of geodesic distances
    # iterate adjacent faces and calculate distances
    for edge, adj_faces in adj_faces_map.items():
        edge = list(edge)
        if len(adj_faces) == 2:
            i = adj_faces[0]
            j = adj_faces[1]

            Gtemp = meshu.geodesic_distance(mesh, i, j, edge)
            Atemp = meshu.angular_distance(mesh, i, j,eta=eta)
            Gval.append(Gtemp)
            Aval.append(Atemp)
            row_indices.append(i)
            col_indices.append(j)
            # add symmetric entry
            Gval.append(Gtemp)
            Aval.append(Atemp)
            row_indices.append(j)
            col_indices.append(i)

        elif len(adj_faces) > 2:
            print("Edge with more than 2 adjacent faces: " + str(adj_faces) + "!")

    Gval = numpy.array(Gval)
    Aval = numpy.array(Aval)
    values = delta * Gval / numpy.mean(Gval) + \
             (1.0 - delta) * Aval / numpy.mean(Aval)

    # create sparse matrix
    distance_matrix = scipy.sparse.csr_matrix(
        (values, (row_indices, col_indices)), shape=(l, l))
    return distance_matrix

def create_affinity_matrix(
    mesh,verbose=False,
    delta = delta_default,
    eta = eta_default):
    """Create the adjacency matrix of the given mesh"""
    st = time.time()
    l = len(mesh.faces)
    if verbose:
        print("mesh_segmentation: Creating distance matrices...")
    distance_matrix = create_distance_matrix(
        mesh,
        delta=delta,
        eta=eta
    )

    if verbose:
        print(f"--> Total time = {time.time() - st}")
        st = time.time()
        print("mesh_segmentation: Finding shortest paths between all faces...")
    # for each non adjacent pair of faces find shortest path of adjacent faces
    W = scipy.sparse.csgraph.dijkstra(distance_matrix)
    inf_indices = numpy.where(numpy.isinf(W))
    W[inf_indices] = 0

    if verbose:
        print(f"--> Total time = {time.time() - st}")
        st = time.time()
        print("mesh_segmentation: Creating affinity matrix...")
    # change distance entries to similarities
    sigma = W.sum()/(l ** 2)
    den = 2 * (sigma ** 2)
    W = numpy.exp(-W/den)
    W[inf_indices] = 0
    numpy.fill_diagonal(W, 1)
    if verbose:
        print(f"--> Total time = {time.time() - st}")
        st = time.time()

    return W

#import time
"""
Conclusion: Didn't do a better job of segmenting the head from the mesh
"""
def segment_mesh(
    mesh,
    k=2,#number of clusters
    delta=0.03, #between 0 and 1
    eta = 0.15,
    action=None, 
    ev_method="sparse",
    kmeans_init='kmeans++',
    verbose = False,):
    """Segments the given mesh into k clusters and performs the given
    action for each cluster
    
    k: number of clusters
    
    delta: closer to zero for more importance on angular distance and 
    closer to 1 for more importance geodesic distance
        default = 0.03,
        min = 0,
        max = 1,
    
    eta (wieght of convexity): the closer to 0 the more important
    the concave angles (min = )
        default = 0.15,
        min = 1e-10,
        max = 1,
    
    ev_method: method for computing the eigenvector decomposition
    """
    st = time.time()
    def _initial_guess(Q, k):
        """Computes an initial guess for the cluster-centers
        Chooses indices of the observations with the least association to each
        other in a greedy manner. Q is the association matrix of the observations.
        """

        # choose the pair of indices with the lowest association to each other
        min_indices = numpy.unravel_index(numpy.argmin(Q), Q.shape)

        chosen = [min_indices[0], min_indices[1]]
        for _ in range(2,k):
            # Take the maximum of the associations to the already chosen indices for
            # every index. The index with the lowest result in that therefore is the
            # least similar to the already chosen pivots so we take it.
            # Note that we will never get an index that was already chosen because
            # an index always has the highest possible association 1.0 to itself
            new_index = numpy.argmin(numpy.max(Q[chosen,:], axis=0))
            chosen.append(new_index)

        return chosen
    
    eta = eta + 1e-9

    # affinity matrix
    
    W = create_affinity_matrix(mesh,verbose=verbose,eta=eta,delta=delta)
    if verbose:
        print("mesh_segmentation: Calculating graph laplacian...")
    # degree matrix
    Dsqrt = numpy.sqrt(numpy.reciprocal(W.sum(1)))
    # graph laplacian
    L = ((W * Dsqrt).transpose() * Dsqrt).transpose()

    if verbose:
        print(f"--> Total time = {time.time() - st}")
        st = time.time()
        print("mesh_segmentation: Calculating eigenvectors...")
    # get eigenvectors
    if ev_method == 'dense':
        _, V = scipy.linalg.eigh(L, eigvals = (L.shape[0] - k, L.shape[0] - 1))
    else:
        _, V = scipy.sparse.linalg.eigsh(L, k)
    # normalize each row to unit length
    V /= numpy.linalg.norm(V, axis=1)[:,None]

    if kmeans_init == 'kmeans++':
        if verbose:
            print(f"--> Total time = {time.time() - st}")
            st = time.time()
            print("mesh_segmentation: Applying kmeans...")
        _, idx = scipy.cluster.vq.kmeans2(V, k, minit='++', iter=50)
    else:
        if verbose:
            print(f"--> Total time = {time.time() - st}")
            st = time.time()
            print("mesh_segmentation: Preparing kmeans...")
        # compute association matrix
        Q = V.dot(V.transpose())
        # compute initial guess for clustering
        initial_centroids = _initial_guess(Q, k)

        if verbose:
            print("mesh_segmentation: Applying kmeans...")
        _, idx = scipy.cluster.vq.kmeans2(V, V[initial_centroids,:], iter=50)

    if verbose:
        print(f"--> Total time = {time.time() - st}")
        st = time.time()
        print("mesh_segmentation: Done clustering!")
    # perform action with the clustering result
    if action:
        action(mesh, k, idx)
        
    return idx

#from python_tools import mesh_utils as meshu



#--- from python_tools ---
from . import ipyvolume_utils as ipvu
from . import numpy_utils as nu

from . import mesh_utils as meshu