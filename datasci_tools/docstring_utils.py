"""
Tutorial 1: Sphinx docstring format

source: https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html#:~:text=The%20Sphinx%20docstring%20format,-In%20general%2C%20a&text=A%20pair%20of%20%3Aparam%3A%20and,values%20returned%20by%20our%20code.

"""

def myfunc(x = 10,y = 20):
    """_summary_

    Args:
        x (int, optional): _description_. Defaults to 10.
        y (int, optional): _description_. Defaults to 20.
    """
    
    
def simple_arg_return_docstring():
    """ function to turn a trimesh object of a neuron into a skeleton, without running soma collapse,
    or recasting result into a Skeleton.  Used by :func:`meshparty.skeletonize.skeletonize_mesh` and
    makes use of :func:`meshparty.skeletonize.skeletonize_components`

    Parameters
    ----------
    mesh: meshparty.trimesh_io.Mesh
        the mesh to skeletonize, defaults assume vertices in nm
    soma_pt: np.array
        a length 3 array specifying to soma location to make the root
        default=None, in which case a heuristic root will be chosen
        in units of mesh vertices
    soma_thresh: float
        distance in mesh vertex units over which to consider mesh 
        vertices close to soma_pt to belong to soma
        these vertices will automatically be invalidated and no
        skeleton branches will attempt to reach them.
        This distance will also be used to collapse all skeleton
        points within this distance to the soma_pt root if collpase_soma
        is true. (default=7500 (nm))
    invalidation_d: float
        the distance along the mesh to invalidate when applying TEASAR
        like algorithm.  Controls how detailed a structure the skeleton
        algorithm reaches. default (10000 (nm))
    smooth_neighborhood: int
        the neighborhood in edge hopes over which to smooth skeleton locations.
        This controls the smoothing of the skeleton
        (default 5)
    large_skel_path_threshold: int
        the threshold in terms of skeleton vertices that skeletons will be
        nominated for tip merging.  Smaller skeleton fragments 
        will not be merged at their tips (default 5000)
    cc_vertex_thresh: int
        the threshold in terms of vertex numbers that connected components
        of the mesh will be considered for skeletonization. mesh connected
        components with fewer than these number of vertices will be ignored
        by skeletonization algorithm. (default 100)
    return_map: bool
        whether to return a map of how each mesh vertex maps onto each skeleton vertex
        based upon how it was invalidated.

    Returns
    -------
        skel_verts: np.array
            a Nx3 matrix of skeleton vertex positions
        skel_edges: np.array
            a Kx2 matrix of skeleton edge indices into skel_verts
        smooth_verts: np.array
            a Nx3 matrix of vertex positions after smoothing
        skel_verts_orig: np.array
            a N long index of skeleton vertices in the original mesh vertex index
        (mesh_to_skeleton_map): np.array
            a Mx2 map of mesh vertex indices to skeleton vertex indices

    """
    
    
    return None

def docstring_with_code_example():
    """Returns a list of cycles which form a basis for cycles of G.

    A basis for cycles of a network is a minimal collection of
    cycles such that any cycle in the network can be written
    as a sum of cycles in the basis.  Here summation of cycles
    is defined as "exclusive or" of the edges. Cycle bases are
    useful, e.g. when deriving equations for electric circuits
    using Kirchhoff's Laws.

    Parameters
    ----------
    G : NetworkX Graph
    root : node, optional
       Specify starting node for basis.

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G.

    Examples
    --------
    >>> G = nx.Graph()
    >>> nx.add_cycle(G, [0, 1, 2, 3])
    >>> nx.add_cycle(G, [0, 3, 4, 5])
    >>> nx.cycle_basis(G, 0)
    [[3, 4, 5, 0], [1, 2, 3, 0]]

    Notes
    -----
    This is adapted from algorithm CACM 491 [1]_.

    References
    ----------
    .. [1] Paton, K. An algorithm for finding a fundamental set of
       cycles of a graph. Comm. ACM 12, 9 (Sept 1969), 514-518.

    See Also
    --------
    simple_cycles
    """