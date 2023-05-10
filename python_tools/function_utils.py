from inspect import getmembers
import inspect
"""
methods for helping inspect functions
"""

#import inspect

def arg_names(func):
    """
    Purpose: To get the names of the argument
    
    from python_tools import function_utils as funcu
    funcu.arg_names(myfunc)
    """
    return inspect.getfullargspec(func).args


#from inspect import getmembers, isfunction

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


def rename(newname):
    """
    Can be used as a decorator to rename the functions
    """
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator



'''
"""
Demo on how to create funcions programmatically,
have to create a wrapper function and call that 
with the parameters you want to signal the 
binding then, otherwise they will do late binding

"""

--Example 1--
my_list = []
for i in range(0,3):
    def make_f(i):
        def f():
            return i
        return f
    my_list.append(make)

--Example 2 ---
def myfunc1(x):
    print("myfunc1") 
def myfunc2(x):
    print("myfunc2") 

new_funcs = []
for f in [myfunc1,myfunc2]:
    def make_func(f):
        def func_wrapper(str1="hi",str2="hello"):
            return str1,str2,f
        return func_wrapper
    new_funcs.append(make_func(f))
    
new_funcs'''

