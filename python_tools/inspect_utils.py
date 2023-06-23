'''

Useful wrappers and functions for the inspect module

'''
import inspect
def getcomments(func):
    return inspect.getcomments(func)

def built_in_func_from_name(obj_name):
    if (("__" == obj_name[:2]) 
        and ("__" == obj_name[-2:])):
        return True
    return False

non_variable_detectors = (
    inspect.isfunction,
    inspect.isclass,
    inspect.ismodule,
    inspect.isbuiltin,
)

def is_global_var_by_value(obj,verbose = False):
    return_value = True
    reason = None
    for k in non_variable_detectors:
        if k(obj):
            reason = k.__name__
            return_value = False
            break
        
    if verbose:
        if reason is None:
            print(f"obj was global variable")
        else:
            print(f"obj not global variable because it {reason}")
            
    return return_value

def global_vars(module,verbose = False,):
    """
    Purpose: Will return the names of the global variables
    defined in the module
    
    from python_tools from . import numpy_utils as nu
    iu.global_vars(nu,verbose = True)
    """
    attr = list(dir(module))
    attr = [k for k in attr if not built_in_func_from_name(k)]
    attr = [k for k in attr if is_global_var_by_value(getattr(module,k))]
    if verbose:
        print(f"# of global variables = {len(attr)}")
    return attr


def function_names(
    module,
    verbose = False):
    """
    Purpose: return all function
    names from module
    """
    results = inspect.getmembers(module,inspect.isfunction)
    results_names = [k[0] for k in results]
    if verbose:
        print(f"# of functions = {len(results_names)}")
        
    return results_names
    
    
def function_code_as_str(func):
    return inspect.getsource(func)
