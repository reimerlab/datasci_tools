
import numpy as np

def projection(
    vector_to_project,
    line_of_projection,
    idx_for_projection=None,
    verbose = False,
    return_magnitude = False):
    
    """
    Purpose: Will find the projection of a vector onto a line
    """
    
    line_of_projection = np.array(line_of_projection)
    vector_to_project = np.array(vector_to_project)
    
    if idx_for_projection is not None:
        vector_to_project = vector_to_project[idx_for_projection]
        line_of_projection = line_of_projection[idx_for_projection]
        
    #2) Find the magnitude of projection of new point onto upward middle vector non scaled
    magn = (vector_to_project@line_of_projection)/(
            line_of_projection@line_of_projection)
    proj_v = magn*line_of_projection
    
    if verbose:
        print(f"magn = {magn}")
        print(f"proj_v = {proj_v}")
        
    if return_magnitude:
        return proj_v,magn
    else:
        return proj_v
    
def error_from_projection(
    vector_to_project,
    line_of_projection,
    idx_for_projection=None,
    verbose = False,):
    
    """
    Purpose: To return the error vector of a projection
    
    Ex: 
    
    lu.error_from_projection(
    vector_to_project=orienting_coords["top_left"],
    line_of_projection = hu.upward_vector_middle_non_scaled,
    verbose = True,
    idx_for_projection=np.arange(0,2)
    )
    """
    
    proj_v = lu.projection(
    vector_to_project=vector_to_project,
    line_of_projection=line_of_projection,
    idx_for_projection=idx_for_projection,
    verbose = verbose,
    return_magnitude = False)
    
    if idx_for_projection is not None:
        error_proj = vector_to_project[idx_for_projection] - proj_v
    else:
        error_proj = vector_to_project - proj_v
    
    if verbose:
        print(f"error_proj = {error_proj}")
        
    return error_proj

def perpendicular_vec_2D(vec):
    return np.array([vec[1],-vec[0]])

def rotation_matrix_2D(angle):
    theta = np.radians(angle)
    r = np.array(( (np.cos(theta), -np.sin(theta)),
               (np.sin(theta),  np.cos(theta)) ))
    return r

    
    
#from python_tools import linalg_utils as lu
    
        




from . import linalg_utils as lu