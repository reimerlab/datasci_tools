"""
methods for helping inspect functions
"""

import inspect

def arg_names(func):
    """
    Purpose: To get the names of the argument
    
    import function_utils as funcu
    funcu.arg_names(myfunc)
    """
    return inspect.getfullargspec(func).args


from inspect import getmembers, isfunction

def all_functions_from_module(
    module,
    return_only_names = False,
    return_only_functions = False):
    
    return_value = list(getmembers(module, isfunction))
    if return_only_names:
        return [k[0] for k in return_value]
    if return_only_functions:
        return [k[1] for k in return_value]
        
    return return_value