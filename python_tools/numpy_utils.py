
from functools import reduce
from pathlib import Path
from pykdtree.kdtree import KDTree
from scipy import stats
from scipy.spatial.distance import pdist,squareform
from shapely.geometry import LineString
from shapely.geometry import Point
import datetime
import itertools
import math
import networkx as nx
from . import numpy_dep as np
import pandas as pd
import scipy.spatial as spatial

import time
import trimesh
#import networkx as nx
#import time
"""
Notes on functionality: 
np.concatenate: combines list of lists into one list like itertools does
np.ptp: gives range from maximum-minimum

np.diff #gets the differences between subsequent elements (turns n element --> n-1 elements)

np.insert(array,indexes of where you want insertion,what you want inserted before the places you specified) --> can do multiple insertions: 

Ex: 
x = np.array([1,4,5,10])
np.insert(x,slice(0,5),2)
>> output: array([ 2,  1,  2,  4,  2,  5,  2, 10])


If want to find the indexes of what is common between 2 1D arrray use
same_ids,x_ind,y_ind = np.intersect1d(soma_segment_id,connectivity_ids,return_indices=True)


"""
def compare_threshold(item1,item2,threshold=0.0001,print_flag=False):
    """
    Purpose: Function that will take a scalar or 2D array and subtract them
    if the distance between them is less than the specified threshold
    then consider equal
    
    Example: 
    nu = reload(nu)

    item1 = [[1,4,5,7],
             [1,4,5,7],
             [1,4,5,7]]
    item2 = [[1,4,5,8.00001],
            [1,4,5,7.00001],
            [1,4,5,7.00001]]

    # item1 = [1,4,5,7]
    # item2 = [1,4,5,9.0000001]

    print(nu.compare_threshold(item1,item2,print_flag=True))
    """
    item1 = np.array(item1)
    item2 = np.array(item2)

    if item1.ndim != item2.ndim:
        raise Exception(f"Dimension for item1.ndim ({item1.ndim}) does not equal item2.ndim ({item2.ndim})")
    if item1.ndim > 2 or item2.ndim > 2:
        raise Exception(f"compare_threshold does not handle items with greater than 2 dimensions: item1.ndim ({item1.ndim}), item2.ndim ({item2.ndim}) ")

    if item1.ndim < 2:
        difference = np.linalg.norm(item1-item2)
    else:
        difference = np.sum(np.linalg.norm(item1 - item2,axis=1))
    
    if print_flag:
        print(f"difference = {difference}")
        
    #compare against threshold and return result
    return difference <= threshold

def concatenate_lists(list_of_lists):
    try:
        return np.concatenate(list_of_lists)
    except:
        return []

#import trimesh
def is_array_like(current_data,include_tuple=False):
    types_to_check = [type(np.ndarray([])),type(np.array([])),list,trimesh.caching.TrackedArray]
    if include_tuple:
        types_to_check.append(tuple)
    return type(current_data) in types_to_check

def non_empty_or_none(current_data):
    if current_data is None:
        return False
    else:
        if len(current_data) == 0:
            return False
        return True

def array_after_exclusion(
                        original_array=[],                    
                        exclusion_list=[],
                        n_elements=0):
    """
    To efficiently get the difference between 2 lists:
    
    original_list = [1,5,6,10,11]
    exclusion = [10,6]
    n_elements = 20

    array_after_exclusion(n_elements=n_elements,exclusion_list=exclusion)
    
    
    ** pretty much the same thing as : 
    np.setdiff1d(array1, array2)

    """
    
    
    if len(exclusion_list) == 0: 
        return original_array
    
    if len(original_array)==0:
        if n_elements > 0:
            original_array = np.arange(n_elements)
        else:
            raise Exceptino("No original array passed")
    else:
        original_array = np.array(original_array)
            
    mask = ~np.isin(original_array,exclusion_list)
    #print(f"mask = {mask}")
    return original_array[mask]

#from pathlib import Path
def load_dict(file_path):
    if file_path == type(Path()):
        file_path = str(file_path.absolute())
      
    my_dict = np.load(file_path,allow_pickle=True)
    return my_dict[my_dict.files[0]][()]


#from scipy.spatial.distance import pdist,squareform
def get_coordinate_distance_matrix(coordinates):
    distance_matrix_condensed = pdist(coordinates,'euclidean')
    distance_matrix = squareform(distance_matrix_condensed)
    return distance_matrix

#import scipy.spatial as spatial
def distance_matrix(
    array1,
    array2=None,
    p=2, #which norm to use
    verbose = False,
    threshold = None,
    default_value = np.inf,
    **kwargs
    ):
    """
    Computes all pairwise distances between 2 arrays
    
    if array1 is M,K  and array2 is N,K
    --> returns M,N array
    
    """
    if array2 is None:
        array2 = array1.copy()
#     if threshold is not None:
#         kwargs["threshold"] = threshold
    st = time.time()
    return_matrix = spatial.distance_matrix(array1,array2,p=p,**kwargs)
    if threshold is not None:
        return_matrix[return_matrix > threshold] = default_value
    if verbose:
        print(f"Total time for distance matrix = {time.time() - st}")
        
    return return_matrix


def get_matching_vertices(possible_vertices,ignore_diagonal=True,
                         equiv_distance=0,
                         print_flag=False):
    """
    ignore_diagonal is not implemented yet 
    """
    possible_vertices = possible_vertices.reshape(-1,3)
    
    dist_matrix = get_coordinate_distance_matrix(possible_vertices)
    
    dist_matrix_copy = dist_matrix.copy()
    dist_matrix_copy[np.eye(dist_matrix.shape[0]).astype("bool")] = np.inf
    if print_flag:
        print(f"The smallest distance (not including diagonal) = {np.min(dist_matrix_copy)}")
    
    matching_vertices = np.array(np.where(dist_matrix <= equiv_distance)).T
    if ignore_diagonal:
        left_side = matching_vertices[:,0]
        right_side = matching_vertices[:,1]

        result = matching_vertices[left_side != right_side]
    else:
        result = matching_vertices
        
    if len(result) > 0:
        return np.unique(np.sort(result,axis=1),axis=0)
    else:
        return result

def number_matching_vertices_between_lists(arr1,arr2,verbose=False):
    stacked_vertices = np.vstack([np.unique(arr1,axis=0),np.unique(arr2,axis=0)])
    stacked_vertices_unique = np.unique(stacked_vertices,axis=0)
    n_different = len(stacked_vertices) - len(stacked_vertices_unique)
    return n_different

def test_matching_vertices_in_lists(arr1,arr2,verbose=False):
    n_different = number_matching_vertices_between_lists(arr1,arr2)
    if verbose:
        print(f"Number of matching vertices = {n_different}")
    if n_different > 0:
        return True
    elif n_different == 0:
        return False
    else:
        raise Exception("More vertices in unique list")

"""
How can find pairwise distance:

example_skeleton = current_mesh_data[0]["branch_skeletons"][0]
ex_skeleton = example_skeleton.reshape(-1,3)


#sk.convert_skeleton_to_graph(ex_skeleton)

#from scipy.spatial.distance import pdist
#import time 
start_time = time.time()
distance_matrix = pdist(ex_skeleton,'euclidean')
print(f"Total time for pdist = {time.time() - start_time}")

returns a matrix that is a lower triangular matrix of size n*(n-1)/2
that gives the distance



"""
def find_matching_endpoints_row(branch_idx_to_endpoints,end_coordinates):
    match_1 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[0]).all(axis=1).reshape(-1,2)
    match_2 = (branch_idx_to_endpoints.reshape(-1,3) == end_coordinates[1]).all(axis=1).reshape(-1,2)
    return np.where(np.sum(match_1 + match_2,axis=1)>1)[0]

def matching_rows_old(vals,row,print_flag=False):

    if len(vals) == 0:
        return np.array([])
    vals = np.array(vals)
    if print_flag:
        print(f"vals = {vals}")
        print(f"row = {row}")
    return np.where((np.array(vals) == np.array(row)).all(axis=1))[0]

def matching_rows(vals,row,
                      print_flag=False,
                      equiv_distance = 0.0001):

    if len(vals) == 0:
        return np.array([])
    vals = np.array(vals)
    row = np.array(row).reshape(-1,3)
    if print_flag:
        print(f"vals = {vals}")
        print(f"row = {row}")
    return np.where(np.linalg.norm(vals-row,axis=1)<equiv_distance)[0]

def matching_row_index(vals,row):
    return matching_rows(vals,row)[0]


# ----------- made when developing the neuron class ------------- #
def sort_multidim_array_by_rows(edge_array,order_row_items=False,):
    """
    Purpose: To sort an array along the 0 axis where you maintain the row integrity
    (with possibly sorting the individual elements along a row)
    
    Example: On how to get sorted edges
    from python_tools import numpy_utils as nu
    nu = reload(nu)
    nu.sort_multidim_array_by_rows(limb_concept_network.edges(),order_row_items=True)
    
    """
    #print(f'edge_array = {edge_array} with type = {type(edge_array)}')
    
    #make sure it is an array
    edge_array = np.array(edge_array)
    
    #check that multidimensional
    if len(edge_array.shape ) < 2:
        print(f"edge_array = {edge_array}")
        raise Exception("array passed did not have at least 2 dimensions")
        
    #will rearrange the items to be in a row if not care about the order here
    if order_row_items:
        edge_array = np.sort(edge_array,axis=1)

    #sort by the x and then y of the egde
    def sorting_func(k):
        return [k[i] for i,v in enumerate(edge_array.shape)]

    #sorted_edge_array = np.array(sorted(edge_array , key=lambda k: [k[0], k[1]]))
    sorted_edge_array = np.array(sorted(edge_array , key=sorting_func))
    
    return sorted_edge_array



def sort_elements_in_every_row(current_array):
    return np.array([np.sort(yi) for yi in current_array])
# --------- Functions pulled from trimesh.grouping ---------- #

def sort_rows_by_column(array,column_idx,largest_to_smallest=True):
    """
    Will sort the rows based on the values of 1 column
    
    """
    order = array[:,column_idx].argsort()
    if largest_to_smallest:
        order = np.flip(order)
    return array[order]

# def sort_rows_by_every_column(array,largest_to_smallest=True):
#     for column_idx in range(array.shape[1]):
#         array =  nu.sort_rows_by_column(array,column_idx,largest_to_smallest=largest_to_smallest)
#     return array

#from functools import reduce

def function_over_multi_lists(arrays,set_function):
    return reduce(set_function,arrays)

def setdiff1d_multi_list(arrays):
    return function_over_multi_lists(arrays,np.setdiff1d)

def logical_and_multi_list(arrays):
    return function_over_multi_lists(arrays,np.logical_and)
def logical_or_multi_list(arrays):
    return function_over_multi_lists(arrays,np.logical_or)

def intersect1d_multi_list(arrays):
    return function_over_multi_lists(arrays,np.intersect1d)

def intersect2d_multi_list(arrays):
    return function_over_multi_lists(arrays,nu.intersect2d)

def union1d_multi_list(arrays):
    return function_over_multi_lists(arrays,np.union1d)

def intersect1d(arr1,arr2,assume_unique=False,return_indices=False):
    """
    Will return the common elements from 2 possibly different sized arrays
    
    If select the return indices = True,
    will also return the indexes of the common elements
    
    
    """
    return np.intersect1d(arr1,arr2,
                         assume_unique=assume_unique,
                         return_indices=return_indices)

def setdiff1d(arr1,arr2,assume_unique=False,return_indices=True):
    """
    Purpose: To get the elements in arr1 that aren't in arr2
    and then to possibly return the indices of those that were
    unique in the first array
    
    
    
    """
    
    arr1 = np.array(arr1)
    leftout = np.setdiff1d(arr1,arr2,assume_unique=assume_unique)
    _, arr_1_indices, _ = np.intersect1d(arr1,leftout,return_indices=True)
    arr_1_indices_sorted= np.sort(arr_1_indices)
    if return_indices:
        return arr1[arr_1_indices_sorted],arr_1_indices_sorted
    else:
        return arr1[arr_1_indices_sorted]
    
def setdiff2d(arr1,arr2):
    try:
        return np.array([k for k in arr1 if len(nu.matching_rows(arr2,k))==0])
    except:
        return np.array([k for k in arr1 if len(nu.matching_rows_old(arr2,k))==0])
    
def intersect2d(arr1,arr2):
    try:
        return np.array([k for k in arr1 if len(nu.matching_rows(arr2,k))>0])
    except:
        return np.array([k for k in arr1 if len(nu.matching_rows_old(arr2,k))>0])
    
def divide_into_label_indexes(mapping):
    """
    Purpose: To take an array that attributes labels to indices
    and divide it into a list of the arrays that correspond to the indices of
    all of the labels
    
    """
    unique_labels = np.sort(np.unique(mapping))
    final_list = [np.where(mapping==lab)[0] for lab in unique_labels]
    return final_list

def turn_off_scientific_notation():
    np.set_printoptions(suppress=True)
    
def average_by_weights(values,weights):
    weights_normalized = weights/np.sum(weights)
    return np.sum(values*weights_normalized)

def angle_between_vectors(v1, v2, acute=True,degrees=True,verbose=False):
    """
    vec1 = np.array([0,0,1])
    vec2 = np.array([1,1,-0.1])
    angle(vec1,vec2,verbose=True)
    """

    dot_product = np.dot(v1, v2)
    if verbose:
        print(f"dot_product = {dot_product}")
    angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    
    if acute == True:
        rad_angle =  angle
    else:
        rad_angle =  2 * np.pi - angle
        
    if degrees:
        return  180* rad_angle/np.pi
    else:
        return rad_angle
            
    
    return return_angle

#import trimesh
def angle_between_vectors_simple(v1,v2):
    return trimesh.geometry.vector_angle([v1,v2])


def intersecting_array_components(arrays,sort_components=True,verbose=False,perfect_match=False):
    """
    Purpose: 
    Will find the groups of arrays that are
    connected components based on overlap of elements
    
    Pseudocode: 
    1) Create an empty edges list
    2) Iterate through all combinations of arrays (skipping the redundants)
    a. heck if there is an intersection
    b. If yes then add to edges list
    3) Trun the edges into a graph 
    4) Return the connected components
    
    """
    
    array_edges = []
    for i,arr1 in enumerate(arrays):
        for j,arr2 in enumerate(arrays):
            if i < j:
                if perfect_match:
                    if len(arr1) != len(arr2):
                        continue
                intersect_elem = np.intersect1d(arr1,arr2)
                if perfect_match:
                    if len(intersect_elem) < len(arr1):
                        continue
                if len(intersect_elem)>0:
                    if verbose:
                        print(f"for edge {[i,j]}, # matching element = {len(intersect_elem)}")
                    array_edges.append([i,j])
                    
                    
    if verbose:
        print(f"array_edges = {array_edges}")
        
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(arrays)))
    G.add_edges_from(array_edges)
    
    conn_comps = list([list(k) for k in nx.connected_components(G)])
    
    if sort_components:
        conn_comps_lenghts = [len(k) for k in conn_comps]
        conn_comps_ordered = [conn_comps[k] for k in np.flip(np.argsort(conn_comps_lenghts))]
        if verbose: 
            print(f"Returning ordered connected components, original lens = {conn_comps_lenghts}")
        conn_comps =  conn_comps_ordered
    
    return np.array(conn_comps)

def array_split(array,n_groups):
    return np.array_split(array,n_groups)

def unique_rows(array):
    return np.unique(array,axis=0)

def unique_non_self_pairings(array):
    """
    Purpose: Will take a list of pairings and 
    then filter the list to only unique pairings where ther is no self
    pairing
    
    
    """
    array = np.array(array)
    
    if len(array) == 0 or (0 in array.shape) :
        return []
    
    array = np.unique(np.sort(np.array(array),axis=1),axis=0)
    array = array[array[:,0] != array[:,1]]
    return array

#import itertools

def choose_k_combinations(array,k):
    return list(itertools.combinations(array,k))

def choose_k_permutations(array,k):
    return list(itertools.permutations(array, k))

def all_unique_choose_2_combinations(array):
    """
    Given a list of numbers  or labels, will 
    determine all the possible unique pariings
    
    """
    starting_node_combinations = list(itertools.combinations(array,2))
    return nu.unique_non_self_pairings(starting_node_combinations)

def all_unique_choose_k_combinations(array,k):
    starting_node_combinations = list(itertools.combinations(array,k))
    return nu.unique_non_self_pairings(starting_node_combinations)
    

    

def unique_pairings_between_2_arrays(array1,array2):
    """
    Turns 2 seperate array into all possible comibnations of elements
    
    [1,2], [3,4]
    
    into 
    
    array([[1, 3],
       [1, 4],
       [2, 3],
       [2, 4]])
    
    
    """
    mesh = np.array(np.meshgrid(array1, array2))
    combinations = mesh.T.reshape(-1, 2)
    return combinations



def remove_indexes(arr1,arr2):
    return np.delete(arr1,arr2)

#from scipy import stats
def mode_1d(array):
    return stats.mode(array)[0][0]
    
def all_subarrays (l): 
    """
    Ex: 
    from python_tools import numpy_utils as nu
    nu.all_subarrays([[1,"a"],[2,"b"],[3,"c"]])
    
    Output:
    [[],
     [[1, 'a']],
     [[2, 'b']],
     [[1, 'a'], [2, 'b']],
     [[3, 'c']],
     [[1, 'a'], [3, 'c']],
     [[2, 'b'], [3, 'c']],
     [[1, 'a'], [2, 'b'], [3, 'c']]]
    """
    base = []   
    lists = [base] 
    for i in range(len(l)): 
        orig = lists[:] 
        new = l[i] 
        for j in range(len(lists)): 
            lists[j] = lists[j] + [new] 
        lists = orig + lists 

    return lists


    
def random_2D_subarray(array,n_samples,
                      replace=False,
                      verbose=False):
    """
    Purpose: To chose a random number of rows from
    a 2D array
    
    Ex: 
    from python_tools import numpy_utils as nu
    from . import numpy_dep as np

    y = np.array([[1,3],[3,2],[5,6]])
    nu.random_2D_subarray(array=y,
                      n_samples=2,
                      replace=False)
    """
    n_samples = int(n_samples)
    if verbose:
        print(f"Sampling {n_samples} rows from array of length {len(array)} with replacement = {replace}")
    random_indexes = np.random.choice(np.arange(len(array)),size=n_samples,replace=replace)
    return array[random_indexes]

random_subarray = random_2D_subarray

def comma_str(num):
    return f"{num:,}"

def array_split(array,n_splits):
    """Split an array into multiple sub-arrays
    
    Ex: 
    from python_tools import numpy_utils as nu
    nu.array_split(np.arange(0,10),3)
    """
    return np.array_split(array,n_splits)


def repeat_vector_down_rows(array,n_repeat):
    """
    Ex: Turn [705895.1025, 711348.065 , 761467.87  ]
        into: 
    
    TrackedArray([[705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ],
              [705895.1025, 711348.065 , 761467.87  ]])
    """
    return np.repeat(array.reshape(-1,3),n_repeat,axis=0)

def all_partitions(array,
    min_partition_size = 2,
    verbose = False):
    """
    Will form all of the possible
    2 partions of an array
    where you can specify the minimum
    number of elements needed 
    for each possible partition
    
    Ex: 
    x = nu.all_partitions(array = np.array([4,5,6,9]))
    """

    choose_k_options = np.arange(min_partition_size,int(len(array)/2)+0.01).astype("int")
    if verbose: 
        print(f"choose_k_options = {choose_k_options}")

    array = np.array(array)

    all_partitions = []
    for k in choose_k_options:
        
        part_1 = nu.choose_k_combinations(array,k)
        part_2 = [np.setdiff1d(array,p1) for p1 in part_1]
        paired_partitions = [[list(u),list(v)] for u,v in zip(part_1,part_2)]

        
        
        if k == len(array)-k:
            paired_partitions = paired_partitions[:int(len(paired_partitions)/2)]

        if verbose:
            #print(f"part_1 = {part_1}")
            #print(f"part_2 = {part_2}")
            for j,pp in enumerate(paired_partitions):
                print(f"partition {j}: {pp}")
                
        all_partitions += paired_partitions

    return all_partitions

#import datetime
def float_to_datetime(fl):
    return datetime.datetime.fromtimestamp(fl)

def obj_array_to_dtype_array(array,dtype=None):
    return np.array(list(array),dtype=dtype)

def save_compressed(array,filepath):
    np.savez_compressed(filepath,data = array)
def load_compressed(filepath):
    return np.load(filepath)["data"]


# ---------- 6/7: Used for synapse filtering -------- #
def indices_of_comparison_func(func,array1,array2):
    """
    Returns the indices of the elements that result
    from applying func to array1 and array2
    """
    return np.nonzero(func(array1, array2))
def intersect_indices(array1,array2):
    """
    Returns the indices of the intersection of array1 and 2
    """
    return indices_of_comparison_func(np.in1d,array1,array2)

def polyval(poly,data):
    return np.polyval(poly,data)

def polyfit(x,y,degree):
    return np.polyfit(x, y, degree)

def weighted_average(array,weights):
    """
    Ex: 
    from python_tools import numpy_utils as nu
    nu.weighted_average(d_widths,d_sk_lengths)
    """
    return np.average(array,weights=weights)


def argnan(array):
    return np.where(np.isnan(np.array(array).astype(float)))[0]

def vector_from_endpoints(start_endpoint,end_endpoint,normalize_vector=True):
    vector = np.array(end_endpoint)-np.array(start_endpoint)
    if normalize_vector:
        vector = vector/np.linalg.norm(vector)
    return vector

def convert_to_array_like(array,include_tuple=False):
    """
    Will convert something to an array
    """
    if not nu.is_array_like(array,include_tuple=include_tuple):
        return [array]
    return array

array_like = convert_to_array_like
def to_list(array):
    return convert_to_array_like(array,include_tuple = True)

def original_array_indices_of_elements(original_array,
                                      matching_array):
    """
    Purpose: Will find the indices of the matching array
    from the original array
    
    Ex: 
    x = [1,2,3,4,5,6]
    y = [4,6,2]
    nu.original_array_indices_of_elements(x,y)
    """
    return np.searchsorted(original_array,
                           matching_array)

def order_arrays_using_original_and_matching(original_array,
                                            matching_array,
                                            arrays,
                                            verbose = False,):
    """
    Purpose: To rearrange arrays so that 
    a specific array matches an original array

    Pseudocode: 
    1) Find the matching array elements
    2) For each array in arrays index using the matching indices
    
    Ex: 
    x = [1,2,3,4,5,6]
    y = [4,6,2]
    arrays  = [ np.array([ "hi","yes","but"])]
    arrays  = [ np.array([ "hi","yes","but"]), ["no","yes","hi"]]
    arrays  = [ np.array([ 1,2,3]), [7,8,9]]
    

    order_arrays_using_original_and_matching(original_array = x,
    matching_array = y,
    arrays=arrays,
    verbose = True)
    
    Return: 
    >>[array(['but', 'hi', 'yes'], dtype='<U3')]
    """
    mapping_indices = nu.original_array_indices_of_elements(original_array,
                                          matching_array)
    sorted_mapping_indices = np.argsort(mapping_indices)
    if verbose:
        print(f"mapping_indices = {mapping_indices}")
        print(f"sorted_mapping_indices = {sorted_mapping_indices}")

    reordered_arrays = [np.array(k)[sorted_mapping_indices] for k in arrays]
    reordered_arrays = [z if type(k) != list else list(z) for k,z in zip(arrays,reordered_arrays)]

    if verbose:
        print(f"reordered_arrays = {reordered_arrays}")

    return reordered_arrays

def order_array_using_original_and_matching(original_array,
                                            matching_array,
                                            array,
                                            verbose = False,):
    """
    Purpose: To rearrange arrays so that 
    a specific array matches an original array

    Pseudocode: 
    1) Find the matching array elements
    2) For each array in arrays index using the matching indices
    
    Ex: 
    x = [1,2,3,4,5,6]
    y = [4,6,2]
    arrays  = [ np.array([ "hi","yes","but"])]
    arrays  = [ np.array([ "hi","yes","but"]), ["no","yes","hi"]]
    arrays  = [ np.array([ 1,2,3]), [7,8,9]]
    

    order_arrays_using_original_and_matching(original_array = x,
    matching_array = y,
    arrays=arrays,
    verbose = True)
    
    Return: 
    >>[array(['but', 'hi', 'yes'], dtype='<U3')]
    """
    mapping_indices = nu.original_array_indices_of_elements(original_array,
                                          matching_array)
    sorted_mapping_indices = np.argsort(mapping_indices)
    if verbose:
        print(f"mapping_indices = {mapping_indices}")
        print(f"sorted_mapping_indices = {sorted_mapping_indices}")

    reordered_array = np.array(array)[sorted_mapping_indices]
    
    if type(array) == list:
        reordered_array = list(reordered_array)
        
    if verbose:
        print(f"reordered_array = {reordered_array}")

    return reordered_array

def divide_data_into_classes(classes_array,data_array,unique_classes=None):
    """
    Purpose: Will divide two parallel arrays of class and the data
    into a dictionary that keys to the unique class and hen 
    all of the data that belongs to that class
    """
    data_array = np.array(data_array)
    if unique_classes is None:
        unique_classes = np.unique(classes_array)
    
    return_dict = dict()
    for c in unique_classes:
        return_dict[c] = data_array[classes_array == c]
        
    return return_dict

def concatenate_arrays_along_last_axis_after_upgraded_to_at_least_2D(arrays):
    """
    Example: 
    from python_tools import numpy_utils as nu
    arrays = [np.array([1,2,3]), np.array([4,5,6])]
    nu.concatenate_arrays_along_last_axis_after_upgraded_to_at_least_2D(arrays)
    
    >> output:
    array([[1, 4],
       [2, 5],
       [3, 6]])
    """
    return np.c_[tuple(arrays)]

def min_max(array,axis=0):
    return np.min(array,axis=axis),np.max(array,axis=axis)
def min_max_3D_coordinates(array):
    return np.array(min_max(array,axis=0))

def bouning_box_corners(array,return_dict = False):
    bbox = min_max_3D_coordinates(array)
    if return_dict and bbox is not None:
        bbox = dict(
            bbox_min_x = bbox[0][0],
            bbox_min_y = bbox[0][1],
            bbox_min_z = bbox[0][2],
            
            bbox_max_x = bbox[1][0],
            bbox_max_y = bbox[1][1],
            bbox_max_z = bbox[1][2],
        )
    return bbox

bounding_box_corners = bouning_box_corners

def bouning_box_midpoint(array):
    return np.mean(nu.bouning_box_corners(array),axis=0)

bounding_box_midpoint = bouning_box_midpoint

def bounding_box_side_lengths(array):
    min_max = nu.min_max(array)
    return min_max[1] - min_max[0]

def bounding_box_volume(array):
    return np.prod(nu.bounding_box_side_lengths(array))

def argsort_multidim_array_by_rows(array,descending=False):
    """
    Ex: 
    x = np.array([
        [2,2,3,4,5],
        [-2,2,3,4,5],
        [3,1,1,1,1],
        [1,10,10,10,10],
        [3,0,1,1,1],
        [-2,-3,3,4,5]
         ])
         
    #showing this argsort will correctly sort
    x[nu.argsort_multidim_arrays_by_rows(x)]
    
    >> Output: 
    
    array([[-2, -3,  3,  4,  5],
       [-2,  2,  3,  4,  5],
       [ 1, 10, 10, 10, 10],
       [ 2,  2,  3,  4,  5],
       [ 3,  0,  1,  1,  1],
       [ 3,  1,  1,  1,  1]])
         
    """
    unique_array = np.unique(array,axis=0)
    argsort_index = np.concatenate([nu.matching_rows_old(array,k) for k in unique_array])
    if descending:
        return argsort_index[::-1]
    else:
        return argsort_index
    
def sort_multidim_array_by_rows(array,descending=False):
    return array[nu.argsort_multidim_array_by_rows(array,
                                                    descending=descending)]

def matrix_of_row_idx(n_rows,n_cols=None):
    if n_cols is None:
        n_cols = n_rows
    return np.repeat(np.arange(0,n_rows).reshape(-1,n_rows).T,n_cols,axis=1)

def matrix_of_col_idx(n_rows,n_cols):
    return matrix_of_row_idx(n_cols,n_rows).T

def argsort_rows_of_2D_array_independently(array,descending=False):
    """
    Purpose: will return array for row idx and one for col idex
    that will sort the values of each row independently of the column
    
    Ex: 
    x = np.array([
        [2,2,3,4,5],
        [-2,2,3,4,5],
        [3,1,1,1,1],
        [1,10,10,10,10],
        [3,0,1,1,1],
        [-2,-3,3,4,5]
         ])
         
    row_idx,col_idx = nu.argsort_rows_of_2D_array_independently(x)
    x[row_idx,col_idx]
    
    Output:
    >>
    array([[ 2,  2,  3,  4,  5],
       [-2,  2,  3,  4,  5],
       [ 1,  1,  1,  1,  3],
       [ 1, 10, 10, 10, 10],
       [ 0,  1,  1,  1,  3],
       [-3, -2,  3,  4,  5]])
    """
    row_idx = nu.matrix_of_row_idx(*array.shape)
    col_idx = np.array([np.argsort(k)[::-1] if descending else np.argsort(k) for i,k in enumerate(array)])
    return row_idx,col_idx



def remove_nans(array):
    array = np.array(array)
    return array[~np.isnan(array)]

def all_directed_choose_2_combinations(array):
    """
    Ex: 
    seg_split_ids = ["864691136388279671_0",
                "864691135403726574_0",
                "864691136194013910_0"]
                
    output: 
    [['864691136388279671_0', '864691135403726574_0'],
     ['864691136388279671_0', '864691136194013910_0'],
     ['864691135403726574_0', '864691136388279671_0'],
     ['864691135403726574_0', '864691136194013910_0'],
     ['864691136194013910_0', '864691136388279671_0'],
     ['864691136194013910_0', '864691135403726574_0']]
    
    """
    combs = []
    for c1 in array:
        for c2 in array:
            if c1 == c2: 
                continue
            combs.append([c1,c2])
            
    return combs


def interpercentile_range(array,range_percentage,axis = None,verbose = False):
    """
    range_percentage should be 50 or 90 (not 0.5 or 0.9)
    
    Purpose: To compute the range that extends from
    (1-range_percentage)/2 to 0.5 + range_percentage/2
    
    Ex: 
    interpercentile_range(np.vstack([np.arange(1,11),
                                np.arange(1,11),
                                np.arange(1,11)]),90,verbose = True,axis = 1)
    """
    lower_perc = (100-range_percentage)/2
    upper_perc = 50 + range_percentage/2
    
    lower_n = np.percentile(array,lower_perc,axis=axis)
    upper_n = np.percentile(array,upper_perc,axis=axis)
    
    interpercentile_range = upper_n - lower_n
    
    if verbose:
        print(f"lower_n = {lower_n} (lower_perc = {lower_perc})")
        print(f"upper_n = {upper_n} (upper_perc = {upper_perc})")
        print(f"interpercentile_range = {interpercentile_range}")
        
    return interpercentile_range

def set_random_seed(seed):
    np.random.seed(seed)
    
    
"""
Can also shuffle with 
idx = np.random.shuffle(idx)
"""
def randomly_shuffle_array(array,**kwargs):
    return np.random.choice(array, len(array), replace=False,**kwargs)


def randomly_sample_array(array,n_samples,replace = True,**kwargs):
    n_samples = int(n_samples)
    return np.random.choice(array,n_samples,replace = replace,**kwargs)
def random_shuffled_indexes_for_array(array,**kwargs):
    idx_to_process = np.arange(0,len(array))
    return nu.randomly_shuffle_array(idx_to_process,**kwargs)

def aligning_matrix_3D(starting_vector,target_vector):
    """
    Derivation from here: 
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d

    
    Will create a matrix that will move the vector starting_vector to be aligned with vector target_vector
    """
    upward_vector = starting_vector
    microns_top_vector = target_vector


    c = microns_top_vector @ upward_vector
    v = np.cross(upward_vector,microns_top_vector)
    
    vp = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

    R = np.eye(3,3) + vp + (vp @ vp)*(1 /(1-c))
    R[:2,:2] = R[:2,:2]*-1
    return R

#from pykdtree.kdtree import KDTree
def closest_dist_between_coordinates(
    array1,
    array2,
    return_min = True,
    return_array_indices = False,
    return_coordinates = False,
    verbose = False,
    ):
    """
    Purpose: To find max/min distance
    betwen two sets of points
    
    Ex: 
    nu.closest_dist_between_coordinates(
        array1 = np.vstack([valid_points,error_points]),
        array2 = test_point,
        return_min=False,
        return_array_indices=True
    )

    """
    array1 = np.array(array1).reshape(-1,3)
    array2 = np.array(array2).reshape(-1,3)

    if return_min:
        f = np.argmin
    else:
        f = np.argmax

    kd = KDTree(np.vstack(array1))
    closest_dist,kd_idx = kd.query(array2)

    array_2_idx = f(closest_dist)
    array_1_idx = kd_idx[array_2_idx]
    dist = closest_dist[array_2_idx]

    if verbose:
        print(f"dist = {dist}")
        print(f"array_1_idx = {array_1_idx}, array_2_idx = {array_2_idx} ")

    if return_coordinates:
        return array1[array_1_idx],array2[array_2_idx]
    
    if return_array_indices:
        return dist,array_1_idx,array_2_idx
    else:
        return dist
    
def closest_idx_for_each_coordinate(
    array,
    array_for_idx,
    closest_idx_algorithm = "kdtree",
    verbose = False,
    return_dists = False,
    ):
    """
    closest_idx,closest_dist = nu.closest_idx_for_each_coordinate(
        np.array([[0,1,0],[0,2,0],[0,3,0]]),
        np.array([[0,1,0],[0,2,0]]),
        return_dists = True
    )
    -> Returns: (array([0, 1, 1], dtype=uint32), array([0., 0., 1.]))
    """
    
    if verbose:
        st = time.time()
        
    array = np.array(array).reshape(-1,3)
    array_for_idx = np.array(array_for_idx).reshape(-1,3)

    if closest_idx_algorithm == "kdtree":
        dist,closest_idx = KDTree(array_for_idx).query(array)
    elif closest_idx_algorithm == "linalg":
        if return_dists:
            dist = closest_idx = np.array([np.min(np.linalg.norm(array_for_idx - k, axis = 1))
                               for k in array])
        closest_idx = np.array([np.argmin(np.linalg.norm(array_for_idx - k, axis = 1))
                               for k in array])
    else:
        raise Exception("")

    if verbose:
        print(f"Total time for closest matching algorithm = {time.time() - st}")
        
    if return_dists:
        return closest_idx,dist
    else:
        return closest_idx

def unravel_index(idx,array_shape):
    return np.unravel_index(idx, array_shape)

def ravel_index(idx,array_shape):
    return np.ravel_multi_index(idx,array_shape)

'''
An example of how to work with the raveling of a matrix

"""
What happens:
1) unravels the z then y then x
"""
x = np.array([[[1,2,3],
              [4,5,6]],
             [[10,20,30],
              [40,50,60]]])
x.ravel()


x_test,y_test,z_test = np.meshgrid(*[np.arange(0,k) for k in x.shape],
                                  indexing="ij")
recov_matrix = x[x_test.ravel(),y_test.ravel(),z_test.ravel()]

print(f"x.ravel = {x.ravel()}")
print(f"recov_matrix = {recov_matrix}")


'''

def all_pairwise_distances_between_coordinates(coordinates):
    """
    Purpose: To get a vector of all the distances between all pairwise coordinates
    """
    coordinates = np.array(coordinates).reshape(-1,3)
    dists = np.triu(nu.get_coordinate_distance_matrix(coordinates)).ravel()
    return dists[dists > 0]


def gram_schmidt_columns(X):
    Q, R = np.linalg.qr(X)
    return Q
def change_basis_matrix(v):
    """
    This just gives change of basis matrix for a basis 
    that has the vector v as its 3rd basis vector
    and the other 2 vectors orthogonal to v 
    (but not necessarily orthogonal to each other)
    *** not make an orthonormal basis ***
    
    -- changed so now pass the non-orthogonal components
    to the QR decomposition to get them as orthogonal
    
    """
    angle = np.pi/2
    rotation_matrix = np.array([[np.cos(angle),-np.sin(angle),0],
                                [np.sin(angle),np.cos(angle),0],
                                [0,0,1]
                               ])
    
    a,b,c = v
    #print(f"a,b,c = {(a,b,c)}")
    if np.abs(c) > 0.00001:
        v_z = v/np.linalg.norm(v)
        v_x = np.array([1,0,-a/c])
        #v_x = v_x/np.linalg.norm(v_x)
        v_y = np.array([0,1,-b/c])
        #v_y = v_y/np.linalg.norm(v_y)
        v_x, v_y = gram_schmidt_columns(np.vstack([v_x,v_y]).T).T
        return np.vstack([v_x,v_y,v_z])
    else:
        #print("Z coeffienct too small")
        #v_z = v
        v[2] = 0
        #print(f"before norm v_z = {v}")
        v_z = v/np.linalg.norm(v)
        #print(f"after norm v_z = {v_z}")
        
        v_x = np.array([0,0,1])
        v_y = rotation_matrix@v_z
        
    return np.vstack([v_x,v_y,v_z])



def bbox_intersect_test_from_corners(
    bbox_corners_1,
    bbox_corners_2,
    verbose = False):
    """
    Purpose: To determine if two bounding boxes
    intersect from their min max corners

    Ex: 
    bbox_intersect_test_from_corners(
    tu.bounding_box_corners(mesh_bbox),
    tu.bounding_box_corners(corners_bbox),
    verbose = True
    )

    """
    bbox_corners_1 = np.array(bbox_corners_1)
    bbox_corners_2 = np.array(bbox_corners_2)
    
    projection_intersect = [not ((bbox_corners_1[0][axis_idx] > bbox_corners_2[1][axis_idx]) or
            (bbox_corners_1[1][axis_idx] < bbox_corners_2[0][axis_idx])) for axis_idx in range(0,3)]
    
    if verbose:
        print(f"projection_intersect = {projection_intersect}")
    
    return nu.logical_and_multi_list(projection_intersect)


def single_thread():
    import os
    os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
    os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
    os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
    os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
    os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

def polar_3D_from_cartesian(x, y=None, z=None):
    """
    Purpose: To change x,y,z cartesian coordinates to 
    polar coordinates: 
    r: distance from origin
    theta:  angle frotated from top of z axis down (helps give the z axis) 
    phi:angle rotation along xy plane from 0 degrees
    
    Ex:
    nu.cart2sph(0.1,0,1)

    """
    if y is None: 
        x_ = x[:,0]
        y = x[:,1]
        z = x[:,2]
        x = x_
    
    xy = np.sqrt(x**2 + y**2) # sqrt(x² + y²)
    
    x_2 = x**2
    y_2 = y**2
    z_2 = z**2

    r = np.sqrt(x_2 + y_2 + z_2) # r = sqrt(x² + y² + z²)

    phi = np.arctan2(y, x) 

    theta = np.arctan2(xy, z) 

    return r, theta, phi
def cartesian_3D_from_polar(
    r,
    theta,
    phi,
    return_array = False,):
    return_value= (
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta)
    )
    
    if return_array:
        return_value = np.vstack(return_value).T.reshape(len(r),3)
    
    return return_value
    
xyz_from_polar = cartesian_3D_from_polar
    

#import math
def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]

def clip(a,a_min,a_max,**kwargs):
    """
    To make sure array or number within limits
    
    Ex: 
    from . import numpy_dep as np
    np.clip(-1,1,9)
    """
    return np.clip(a,a_min,a_max,**kwargs)

def str_repr_of_array_to_array(array):
    return np.array(eval(array))

def reject_outliers(array, m=2,return_mask = False):
    mask = abs(array - np.mean(array)) < m * np.std(array)
    if return_mask:
        return mask
    return array[mask]

#import itertools
def binary_permutation_matrix(n):
    """
    Purpose: to get a binary matrix of n number of variables
    showing all the possible binary combinations
    """
    lst = np.array(list(itertools.product([0, 1], repeat=n)))
    return lst

def interval_bins_covering_array(
    array,
    n_intervals,
    overlap = 0,
    outlier_buffer = 0,
    verbose = False,
    ):
    """
    Create a set of intervals with a certain amount
    of overlap in between each 
    
    Ex: 
    from python_tools import numpy_utils as nu
    nu.interval_bins_covering_array(
        array = interval_vals,
        n_intervals = 10,
        overlap = 20, #if this is a percentage then it is a proportion of the interval
        outlier_buffer = 0,
        verbose = False,
    )
    """

    if outlier_buffer is not None and outlier_buffer > 0:
        original_size = len(array)
        array = array[(array >= np.percentile(array,outlier_buffer)) &
                      (array <= np.percentile(array,100-outlier_buffer))]

        if verbose:
            print(f"After interval outlier filtering array reduced from {original_size} to {len(array)} entries")

    array_min = np.min(array)
    array_max = np.max(array)
    array_size = array_max - array_min
    if verbose:
        print(f"array_min = {array_min}, array_max = {array_max}")


    if overlap == 0:
        boundaries = np.linspace(array_min,array_max,n_intervals + 1)
        intervals = np.vstack([boundaries[:-1],boundaries[1:]]).T
        interval_size = intervals[0][1] - intervals[0][0]
    if overlap > 0:
        if overlap < 1:
            overlap = array_size * overlap

        if verbose:
            print(f"Overlap size = {overlap}")

        interval_size = ((array_size) + overlap*(n_intervals - 1))/n_intervals
        if verbose:
            print(f"interval_size = {interval_size}")


        if overlap > interval_size:
            raise Exception(f"Overlap size {overlap} is greater than {interval_size}")

        non_overlap_size = interval_size - overlap
        boundaries_left = np.arange(0,n_intervals)*non_overlap_size
        boundaries_right = boundaries_left + interval_size
        intervals = array_min + np.vstack([boundaries_left,boundaries_right]).T

    if verbose:
        print(f"intervals = {intervals}")
        
    return intervals

def ind_conversion(
    idx,
    shape,
    order_start,
    ):
    """
    Purpose: convert flattened indices
    from one order to another (example
    from C to Fortran or Fortran to C)
    
    Ex:
    image_height=3
    image_width=4
    mp_ = np.array([ 3,  6,  1,  4, 10,  2])

    ind_conversion(
       idx=mp_,
        shape = (image_height,image_width),
        order_start = "f"
    )
    
    >> output: mp_=np.array([1,2,4,5,7,8]) 
    """
    idx = np.array(idx).astype("int")
    order_start = order_start.lower()
    if order_start == "c":
        order_end = "f"
    elif order_start == "f":
        order_end = "c"
    else:
        raise Exception(f"Unknown order {order_start}")
        
    coords=np.unravel_index(idx, shape, order=order_start.upper())  
    ind_out = np.ravel_multi_index(coords, shape, order=order_end.upper())
    
    return ind_out

def arange_with_leftover(stop,start=0,step = 1,dtype=None,tol=0.0001):
    """
    Purpose: to do what arange does but include the
    little part that might be left over
    
    Ex: 
    from python_tools import numpy_utils as nu
    nu.arange_with_leftover(10.1)
    """
    curr_array = np.arange(stop=stop,start=start,step=step,dtype=dtype)
    if (np.abs(curr_array[-1]-stop) > tol):
        curr_array = np.hstack([curr_array,[stop]])
        
    return curr_array

#import itertools
def all_combinations_of_lists(*args):
    """
    Purpose: Given a list of attributes and their possible attribute
    values, find all the unique combinations of all the lists
    
    Ex: 
    all_combinations_of_lists(
        [1,5,6],
        ["hi","hello","yes"],
        [True,False],
    )
    """
    return list(itertools.product(*args))

def is_list_of_lists(array):
    return any(isinstance(el, list) for el in array) or any(isinstance(el, np.array) for el in array)

def coordinates_edges_from_line_segments(array):
    """
    Purpose: will return the nodes and edges but without making the nodes unique
    (so will have some repeats)
    
    """
    unique_rows  = np.array(array).reshape(-1,3)
    curr_edges = np.arange(len(unique_rows)).reshape(-1,2)
    return unique_rows, curr_edges

def bounding_box(array):
    array = np.array(array).reshape(-1,3)
    return np.vstack([np.min(array,axis=0),
                     np.max(array,axis=0)])


    
def edge_list_from_adjacency_matrix(
    array,
    add_self_loops=False,
    bidirectional = False,):
    return_edges = np.vstack(np.where(array == 1)).T.reshape(-1,2)
    if bidirectional and len(return_edges) > 0:
        return_edges = np.vstack([return_edges,return_edges[:,[1,0]]])
    if add_self_loops:
        return_edges = np.vstack([return_edges.reshape(-1,2),xu.self_loop_edges(len(array))])
    return return_edges.astype('int')

def adjacency_matrix_from_edge_list(array):
    """
    THIS IS ASSUMING THAT ALL NODES IN THE GRAPH HAVE AT LEAST ONE EDGE
    """
    max_index = np.max(array.ravel())
    empty_array = np.zeros((max_index+1,max_index+1)).astype('int')
    empty_array[array.T[0],array.T[1]] = 1
    return empty_array

def replace_nan_with_zero(array):
    return np.nan_to_num(array)

def log_n(array,base):
    return np.log(array) / np.log(base) 

def is_inf(x):
    return  ~np.isfinite(x)

def is_nan(x):
    return np.isnan(x)

def is_nan_or_inf(x):
    return nu.is_nan(x) | nu.is_inf(x) 


def example_apply_along_axis():
    x = np.arange(0,15).reshape(-1,3)
    weights = np.array([1,2,3])
    np.apply_along_axis(nu.weighted_average,axis=1,arr=x,weights=weights)
    
def weighted_average_along_axis(array,weights,axis=0):
    return np.apply_along_axis(nu.weighted_average,axis=axis,arr=array,weights=weights)
    
def dict_map_array(array,map):
    return np.vectorize(map.get)(array)

def angle_from_xy_vec(xy):
    """
    Purpose: To compute angles from the vectors
    
    Ex
    nu.angle_from_xy_vec(np.array([[1,0],[0,1],[-1,0]]).T)
    """
    if xy.shape[0] != 2:
        xy = xy.T
    angle = np.arctan2(xy[1],xy[0])
    return np.degrees(angle) % 360.0  


#from scipy import stats
def equal_depth_bins(array,n_bins=10):
    array = remove_nans(array)
    return stats.mstats.mquantiles(array, np.linspace(0,1,n_bins+1))

def equal_width_bins(array,n_bins = 10):
    array = remove_nans(array)
    return  np.linspace(np.min(array),np.max(array),n_bins+1)

def bin_array(array,n_bins = 10,bin_type = "equal_width"):
    array = remove_nans(array)
    return getattr(nu,f"{bin_type}_bins")(array,n_bins = n_bins)
    

#from scipy import stats
def mode(array,axis=0,return_counts = False):
    """
    Purpos: To get the mode of an array
    """
    m = stats.mode(array,axis=axis)
    
    mode = m.mode.squeeze()
    
    if return_counts:
        return mode,m.count.squeeze()
    else:
        return mode
    
#from shapely.geometry import LineString
#from shapely.geometry import Point

def circle_intersect_by_line_semgment(
    array,
    circle_center,
    circle_radius,
    verbose = False,
    ):

    """
    Purpose: To find the intersection points of a 
    circle and a line
    
    Ex: 
    from python_tools import numpy_utils as nu
    nu.circle_intersect_by_line_semgment(
        array = np.array([(0,0), (10,10)]),
        circle_center=(5,5),
        circle_radius=3
    )
    """
    dim = array.shape[-1]
    p = Point(*circle_center)
    c = p.buffer(circle_radius).boundary
    l = LineString(array)
    i = c.intersection(l)


    if i.geom_type == 'MultiPoint':
        intersect = np.array([k.coords[0] for k in list(i.geoms)]).reshape(-1,dim)
    elif i.geom_type == "Point":
        intersect = np.array([i.coords[0]]).reshape(-1,dim)
    else:
        intersect = np.array([]).reshape(-1,dim)

    return intersect
    
def repeat_vector_hstack(array):
    return np.tile(array.T,(2,1)).T

def angle_from_chord(chord,radius,rad = True):
    """
    Purpose: To return angle in a circle defined by the
    length of the chord it forms with a circle of raidus r
    """
    return_value = 2*np.arcsin(chord/(2*radius))
    if not rad:
        return_value = return_value * 180/ np.pi
        
    return return_value


# ---------- circular statistics -----
def cdiff(alpha, beta, period=2*np.pi,rad=True,):
    """
    Returns the cirvular difference between two orientations given the period
    
    Ex: 
    from python_tools import numpy_utils as nu
    nu.cdist(180,20,rad=False)
    """
    if not rad:
        if period <= np.pi*2:
            period = period * 180 / np.pi
    return (alpha - beta + period / 2) % period - period / 2


def cdist(alpha, beta, period=2*np.pi,rad=True):
    """
    Returns the cirvular distance between two orientations given the period

    Example:
        from . import numpy_dep as np
        from matplotlib import pyplot as plt
        ori_scale = np.linspace(0, np.pi, 100)
        ori_x, ori_y = np.meshgrid(ori_scale, ori_scale)
        delta_ori = cdist(ori_x, ori_y)
        plt.scatter(ori_x.ravel(), ori_y.ravel(), c=delta_ori.ravel())
        plt.colorbar()
    """
    return np.abs(cdiff(alpha, beta, period,rad=rad))


def bbox_with_buffer(
    bbox=None,
    center = None,
    buffer_array = None,
    percentage = None,
    buffer = 0,
    buffer_x = None,
    buffer_y = None,
    buffer_z = None,
    buffer_x_min = None,
    buffer_x_max = None,
    buffer_y_min = None,
    buffer_y_max = None,
    buffer_z_min = None,
    buffer_z_max = None,
    multiplier = None,
    verbose = False,
    subtract_buffer = True,
    return_buffer_array = False,
    ):
    """
    To create a bounding box with a buffer either
    subtracted or added to it
    """
    #print(f"buffer_array = {buffer_array}")
    if (percentage is not None) and buffer_array is None:
        if percentage == True:
            percentage = None
        buffer_array = (bbox[1] - bbox[0]) * (percentage/100)

    #print(f"buffer_array = {buffer_array}")
    if buffer_array is not None:
        if buffer_array.ndim == 1 or (buffer_array.ndim == 2 and len(buffer_array)==1):
#                 print(f"Replacing buffer_array")
#                 print(f"buffer_array.ndim = {buffer_array.ndim}, len(buffer_array) = {len(buffer_array)}")
            buffer_array = np.vstack([buffer_array,buffer_array])
        buffer_x_min,buffer_y_min,buffer_z_min = buffer_array[0]
        buffer_x_max,buffer_y_max,buffer_z_max = buffer_array[1]
    else:
        if buffer_x is None:
            buffer_x = buffer

        if buffer_y is None:
            buffer_y = buffer

        if buffer_z is None:
            buffer_z= buffer

        if buffer_x_min is None:
            buffer_x_min = buffer_x
        if buffer_x_max is None:
            buffer_x_max = buffer_x
        if buffer_y_min is None:
            buffer_y_min = buffer_y
        if buffer_y_max is None:
            buffer_y_max = buffer_y
        if buffer_z_min is None:
            buffer_z_min = buffer_z
        if buffer_z_max is None:
            buffer_z_max = buffer_z

        
    buffer_array = np.array([[buffer_x_min,buffer_y_min,buffer_z_min],
                              [buffer_x_max,buffer_y_max,buffer_z_max]])
    
    if multiplier is not None:
        buffer_array = buffer_array * buffer_array_multipier
    
    if return_buffer_array:
        return buffer_array
    
    if verbose:
        print(f"Original bbox:\n{bbox}")
        
    subtract_array = np.array([[buffer_x_min,buffer_y_min,buffer_z_min],
                              [-buffer_x_max,-buffer_y_max,-buffer_z_max]])
    
    if not subtract_buffer:
        subtract_array = -1* subtract_array
    #print(f"subtract_array = {subtract_array}")

    if None in subtract_array:
        raise Exception("")
        
    if bbox is None:
        bbox = np.vstack([center,center])

    new_bounds = bbox + subtract_array
    if verbose:
        print(f"New bbox:\n{new_bounds}")
    return new_bounds

def bbox_from_center_and_widths(
    center,
    buffer = None,
    buffer_x = None,
    buffer_y = None,
    buffer_z = None,
    ):
    
    return nu.bbox_with_buffer(
    center = center,
    buffer = buffer,
    buffer_x = buffer_x,
    buffer_y = buffer_y,
    buffer_z = buffer_z,)

def bbox_cube_from_bbox(
    bbox,
    verbose = False,
    ):
    """
    Purpose: To create a bounding box  that
    is square based on the longest current side 
    length of the bounding bbox
    """
    axes_bbox = bbox
    axes_bbox_center = np.mean(axes_bbox,axis = 0)
    max_side_length = np.max(np.abs(axes_bbox[0,:] - axes_bbox[1,:]))

    if verbose:
        print(f"max_side_length = {max_side_length}")

    new_bbox = nu.bbox_from_center_and_widths(
        axes_bbox_center,
        buffer = max_side_length/2
    )

    return new_bbox[[1,0],:]


def angle_between_matrix_of_vectors_and_vector(
    array,
    vector,
    acute=True,
    degrees=True):
    
    dot_product = array @ np.expand_dims(vector,axis=1).ravel()
    angle = np.arccos(dot_product / (np.linalg.norm(array,axis=1) * np.linalg.norm(vector)))
    
    if acute == True:
        rad_angle =  angle
    else:
        rad_angle =  2 * np.pi - angle
        
    if degrees:
        return  180* rad_angle/np.pi
    else:
        return rad_angle
    
    
def mask_of_array_1_elements_in_array_2(
    array1,
    array2,
    ):
    """
    Purpose: To return a boolean mask of size
    array1.shape that indicates which elements
    of array1 are in array2
    
    Ex: 
    x = np.array([1,2,3,4,5,6])
    y = np.array([1,4,8,9,0,10,10,10,10,10])

    >> output: array([ True, False, False,  True, False, False])
    """
    
    return np.in1d(array1,array2)


# --- computing eigenvalues ----
"""
Note: 
eigh: eigenvalues are sorted, but only works for symmetric matricies

eig: works on non-symmetric matricies, but eigenvalues not sorted
"""
def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)
is_symmetric = check_symmetric

def eig_vals_vecs(
    array,
    verbose = True):
    
    if not check_symmetric(array):
        raise Exception("")
    
    eigvals,eigvecs = np.linalg.eigh(array.astype('float'))
    return eigvals,eigvecs 

eig_vals_vecs_of_symmetric = eig_vals_vecs

def eig_vals_vecs_of_nonsymmetric(
    array,
    verbose = False):
    if verbose:
        print(f"**warning: eigenvalues are not ordered")
        print(f"Columns arae eigenvectors")
    eigvals,eigvecs = np.linalg.eig(array.astype('float'))
    
    return eigvals,eigvecs 

def eigenvalue_max(array):
    return np.max(np.linalg.eig(array)[0])
def eigenvalue_max_sym(array):
    return np.max(np.linalg.eigh(array)[0])
def eigenvalue_min(array):
    return np.min(np.linalg.eig(array)[0])

def diagonal_vector_from_array(array):
    return np.diag(array)
def diagonal_matrix_from_array(array):
    return np.diag(np.diag(array))


def rows_columns_restriction(
    array,
    rows,
    columns,
    filter_away = False):
    """
    Purpose: To select only certain
    rows and columns from an array
    
    Ex: 
    a = np.arange(20).reshape((5,4))
    nu.rows_columns_restriction(
        a,
        rows = [0,1,3],
        columns = [0,2]
    )
    
    """
    if filter_away:
        rows = np.delete(np.arange(array.shape[0]),rows)
        columns = np.delete(np.arange(array.shape[1]),columns)
    return array[np.ix_(rows,columns)]
    
def rows_columns_delete(
    array,
    rows,
    columns,
    ):
    """
    Purpos: To delete rows and columns
    from an array
    
    a = np.arange(20).reshape((5,4))
    nu.rows_columns_delete(
        a,
        rows = [0,1,3],
        columns = [0,2],

    )
    """
    return rows_columns_restriction(
    array,
    rows,
    columns,
    filter_away = True)

        
#import pandas as pd
#from python_tools import pandas_utils as pu
def array_of_coordinates_and_labels_from_dict(
    coordinate_dict,
    label_name = "label",
    attributes_dict = None,
    return_df = False):
    """
    Purpose: to convert a dictionary with a label
    mapping to coordinates to an array of coordinaates
    and an array of labels (can export df as well)
    """

    
    all_comp_scatters_comb = {k:np.vstack(v) for k,v in coordinate_dict.items()}
    
        
    if len(all_comp_scatters_comb) > 0:
        all_comp_scatters_array = np.vstack(list(all_comp_scatters_comb.values()))
        compartment_labels = np.hstack([[k]*len(v) for k,v in all_comp_scatters_comb.items()])
        
    else:
        all_comp_scatters_array = np.array([])
        compartment_labels = np.array([])

    if return_df:
        if len(all_comp_scatters_array) > 0:
            df = pd.DataFrame(all_comp_scatters_array)
            df.columns = ["x","y","z"]
            df[label_name] = compartment_labels
            return_value = df
        else:
            return_value = pu.empty_df()
    else:
        return_value = all_comp_scatters_array,compartment_labels

    return return_value
    
def df_of_coordinates_and_labels_from_dict(
    coordinate_dict,
    label_name = "label",
    ):
    
    return nu.array_of_coordinates_and_labels_from_dict(
        coordinate_dict,
        label_name = label_name,
        return_df = True)

def mean_coordinates_from_radius_threshold_clustering(
    array,
    radius,
    verbose = False,
    return_clustering_idx = False,):
    """
    Purpose: To reduce a set of coordinates to 
    the mean of the cluster of coordinates where
    coordinates are clustered together if within
    a certain threshold distance of another in the group

    Pseudocode: 
    1) get a radius threshold graph of the coordinates
    2) Divide nodes of graph into connected components
    3) For each connected component compute the mean coordinate
    4) Stack coordinates
    """

    G_closest = xu.radius_threshold_graph_from_coordinates(
        coordinates = array,
        radius = radius,
    )
    
    conn_comp = xu.connected_components(G_closest)
    
    if verbose:
        print(f"# of conn_comp = {len(conn_comp)}")
        
    mean_coordinates = np.array([
        np.mean(array[c].reshape(-1,3),axis=0)
        for c in conn_comp]).reshape(-1,3)
    
    if verbose:
        print(f"mean_coordinates = {mean_coordinates}")
        
    if return_clustering_idx:
        return mean_coordinates,conn_comp
    else:
        return mean_coordinates
    
def grid_array(xmin,xmax,ymin,ymax,zmin=None,zmax = None,n_intervales = 200):
    if zmin is None:
        return np.mgrid[xmin:xmax:n_intervales*1j, ymin:ymax:n_intervales*1j]
    else:
        return np.mgrid[xmin:xmax:n_intervales*1j, ymin:ymax:n_intervales*1j, zmin:zmax:n_intervales*1j]
    
def rotation_matrix(degrees):
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def histogram(array,bins = 10,**kwargs):
    return np.np.histogram(array,bins = bins,**kwargs)

def array_size(array):
    return array.size * array.itemsize

def random_idx(
    n_samples=None,
    array = None,
    array_len = None,
    seed = None,
    samples_perc = None,
    replace = False,
    verbose = False,
    **kwargs):
    
    if seed is not None:
        np.random.seed(seed)
    if array_len is None:
        array_len = len(array)
        
    if n_samples is None:
        n_samples = np.ceil(array_len*samples_perc/100).astype('int')
        
    n_samples = int(n_samples)
    if verbose:
        print(f"Sampling {n_samples} rows from array of length {len(array)} with replacement = {replace}")
 
    random_indexes = np.random.choice(np.arange(array_len),size=n_samples,replace=replace)
    return random_indexes

def empty_n_by_m_default_matrix(n,m=None,default_value = None):
    if m is None:
        m = n
    return np.repeat([default_value]*m,n).reshape(n,m)


def axes_max_radius(array):
    lims = np.vstack([
        np.min(array,axis=0),
        np.max(array,axis=0)
    ])

    radii = np.abs(
        lims[1,:] - lims[0,:],
    )/2

    max_radii = np.max(radii)
    return max_radii
    
def axes_lim_equal_width(
    array,
    buffer=0,
    flip_y = False,
    verbose = False,
    ):
    """
    Purpose: calculate axes limits
    so that the scaling is consistent across
    all axes (all axes limits have same width)
    """

    lims = np.vstack([
        np.min(array,axis=0),
        np.max(array,axis=0)
    ])

    radii = np.abs(
        lims[1,:] - lims[0,:],
    )/2

    max_radii = np.max(radii)

    midpoint = lims.mean(axis=0)

    if flip_y:
        midpoint[1] = -1*midpoint[1] 

    axes_lims  = np.vstack([
        [-1*(max_radii + buffer)]*lims.shape[-1],
        [1*(max_radii + buffer)]*lims.shape[-1],
    ]
    )
    new_lim = axes_lims + midpoint


    if verbose:
        print(f"midpoint = {midpoint}")
        print(f"new_lim = {new_lim}")

    return new_lim

def midpoints_from_array(arr):
    """
    Purpose: To find the midpoints between neighbors
    in an array
    """
    return np.mean(np.vstack([arr[:-1],arr[1:]]),axis = 0)

def bin_ndim_array(
    array,
    n_bins = 10,
    bin_width = None,
    return_bin_centers =False,
    eta = 0.00001,
    verbose = False,
    ):
    """
    Purpose: Want to bin data in n dimensions
    (for each datapoint have an n size vector
    that has n indexes of bins in each direction)

    Can return the bin centers as well

    Pseudocode: 
    a_idx,a_mid = nu.bin_ndim_array(
        array = leaf_df_with_vec[pu.coordinate_columns("centroid")].to_numpy(),
        bin_width=100_000,
        return_bin_centers=True,
        verbose = False
    )
    """
    if bin_width is not None:
        if not nu.is_array_like(bin_width):
            bin_width = [bin_width]*(array.shape[-1])
        n_bins = [None]*(array.shape[-1])
    else:
        bin_width = [None]*(array.shape[-1])

    if "int" in str(type(n_bins)):
        n_bins = [n_bins]*(array.shape[-1])

    bins_by_axes = []
    arr_idx = np.zeros(array.shape).astype('int')
    for j,(nb,bw,arr) in enumerate(zip(n_bins,bin_width,array.T)):
        if nb is None:
            dist = arr.max() - arr.min()
            nb = np.ceil(dist/bw).astype('int')
        if verbose:
            print(f"Axis {j}: nbins = {nb}")

        curr_bins = np.linspace(arr.min(),arr.max() + eta,nb+1)
        bins_by_axes.append(curr_bins)
        arr_idx[:,j] = np.digitize(arr,curr_bins) - 1

        
    if return_bin_centers:
        # now to get the bin midpoints
        bins_mid_by_axes = [nu.midpoints_from_array(ba) for ba in bins_by_axes ]

        arr_idx_mid = np.vstack([
            bmid[arri] for bmid,arri in zip(bins_mid_by_axes,arr_idx.T)
        ]).T
        return arr_idx,arr_idx_mid

    return arr_idx

def osi_from_directions(directions,verbose = False):
    """
    Purpose: Orientational Selectivity Index (between 0 and 1)
    """
    f1 = np.exp(directions / 90 * np.pi * 1j).sum()
    f0 = len(directions)
    osi = np.abs(f1) / f0
    if verbose:
        print(f"osi = {osi}")
    return osi

def dsi_from_directions(directions,verbose = False):
    """
    Purpose: Directional Selectivity Index (between 0 and 1)
    """
    f1 = np.exp(directions / 180 * np.pi * 1j).sum()
    f0 = len(directions)
    dsi = np.abs(f1) / f0
    if verbose:
        print(f"dsi = {dsi}")
    return dsi

def normalize_axis(array,axis = -1):
    return array/np.linalg.norm(array,axis = axis).reshape(-1,1)

def min_unique_pairing(
    array_1,
    array_2,
    closest_idx_algorithm = "linalg",
    verbose = False,
    return_dist = False,
    ):
    """
    Purpose: Want to find the closest idx
    for each coordinate where no 
    2 coordinates can be paired up to the
    same idx

    Pseudocode: 
    0) Create a mask of those already paired and 
    those paired (set all as false)
    1) Iterate through number of points to pair
    -- 
    a) Compute the shortest dist to all points
    (not already paired)
    b) Get the min distance and add the pairing 
    to the mask and paired list
    c) Continue until all are paired or have 
    been paired to


    """
    array = array_1
    array_for_idx = array_2
    
    idx_mask = np.arange(array_for_idx.shape[0])
    array_mask = np.arange(array.shape[0])
    pairings = []
    pairings_dist = []
    

    for i in range(len(array_mask)):
        if len(array_mask) == 0 or len(idx_mask) == 0:
            break
        closest_idx,closest_dist = nu.closest_idx_for_each_coordinate(
            array[array_mask],
            array_for_idx[idx_mask],
            closest_idx_algorithm=closest_idx_algorithm,
            return_dists = True
        )

        idx = np.argmin(closest_dist)
        pairings_dist.append(closest_dist[idx])
        array_win = array_mask[idx]
        idx_win = idx_mask[closest_idx[idx]]
        pairings.append([array_win,idx_win])
        
        if verbose:
            print(f"pair {i}: {(array_win,idx_win)} (dist = {closest_dist[idx]})")

        array_mask = array_mask[array_mask!= array_win]
        idx_mask = idx_mask[idx_mask != idx_win]

    if return_dist:
        return pairings,pairings_dist
    else:
        return pairings
    
    
def loadtxt(filepath,dtype = "float",delimiter=" ",**kwargs):
    filepath = str(Path(filepath).absolute())
    return np.loadtxt(filepath,dtype = dtype,delimiter = delimiter,**kwargs)

def remove_range_list(
    obj,
    range_list,
    remove = True,
    verbose = False):
    """
    Purpose: To select or remove a list of ranges
    from string

    Source: https://www.geeksforgeeks.org/python-remove-index-ranges-from-string/

    Ex:
    remove_range_list(
        obj = 'geeksforgeeks is best for geeks',
        range_list = [(3, 6), (7, 10), (14, 17)],
        remove = True,
        verbose = True,
    )
    """
    if verbose:
        print(f"Original obj: {obj}")
    range_list = nu.to_list(range_list)

    res = ""

    indices = [remove] * len(obj)

    for start_idx, end_idx in range_list:
        for i in range(start_idx, end_idx):
            indices[i] = not remove

    res = ''.join(itertools.compress(obj, indices))

    # printing result
    if verbose:
        print("The reconstructed obj : " + str(res))

    return res

def keep_range_list(
    obj,
    range_list,
    verbose = False):
    
    return remove_range_list(
        obj=obj,
        range_list=range_list,
        remove = False,
        verbose = verbose
    )
    
def row_col_subarray(arr,row_idx,col_idx):
    """
    Purpose: Will select a 2D subarray
    after given the columns and rows to keep
    
    Example of similar operation
    import numpy as np
    test_arr = np.array([
        [1,2,3],
        [3,4,5],
        [6,7,8]
    ])

    row_idx = [0,2]
    col_idx = [0,1]
    test_arr[row_idx,:][:,col_idx]
    """
    row_idx = to_list(row_idx)
    col_idx = to_list(col_idx)
    return arr[row_idx, :][:, col_idx]

array_from_txt = loadtxt
read_txt = loadtxt



#--- from python_tools ---
from . import networkx_utils as xu
from . import pandas_utils as pu

from . import numpy_utils as nu