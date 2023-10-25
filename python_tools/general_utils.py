
import itertools
from . import numpy_dep as np

"""
How to recieve a single values tuple 
and unpack it into its element --> use (__ , ) on recieving end

def ex_function(ex_bool = True,second_bool=False):
    
    
    return_value = (5,)
    if ex_bool:
        return_value += (6,)
    if second_bool:
        return_value += (7,)
        

    return return_value

(y,) = ex_function(ex_bool=False,second_bool=False)
type(y)



To us eval in list comprehension: 

globs = globals()
locs = locals()
out2 = [eval(cmd,globs,locs) for cmd in ['self.b']]

"""


#from datasci_tools import numpy_utils as nu
#from . import numpy_dep as np
#import itertools
def invert_mapping(my_map,total_keys=None,one_to_one=False):
    """
    Will invert a dictionary mapping that is not unique
    (Also considers an array of a mapping of the indices to the value)
    
    Ex: 
    input: [8,1,4,5,4,6,8]
    output: {8: [0, 6], 1: [1], 4: [2, 4], 5: [3], 6: [5]}
    """
    if type(my_map) == dict:
        pass
    elif nu.is_array_like(my_map):
        my_map = dict([(i,k) for i,k in enumerate(my_map)])
    else:
        raise Exception("Non dictionary or array type recieved")
        
    if total_keys is None:
        inv_map = {}
    else:
        inv_map = dict([(k,[]) for k in total_keys])
    
    #handling the one-dimensional case where dictionary just maps to numbers
    if np.isscalar(list(my_map.values())[0]):
        for k, v in my_map.items():
            inv_map[v] = inv_map.get(v, []) + [k]
    else: #2-D case where dictionary maps to list of numbers
        for k,v1 in my_map.items():
            for v in v1:
                inv_map[v] = inv_map.get(v, []) + [k]
    
    if one_to_one:
        inv_map = {k:v[0] for k,v in inv_map.items()}
        
    return inv_map

def get_unique_values_dict_of_lists(dict_of_lists):
    """
    Purpose: If have a dictionary that maps the keys to lists,
    this function will give the unique values of all the elements of all the lists
    
    
    """
    return set(list(itertools.chain.from_iterable(list(dict_of_lists.values()))))

def flip_key_orders_for_dict(curr_dict):
    """
    To flip the order of keys in dictionarys with multiple
    levels of keys:
    
    Ex: 
    test_dict = {0:{1:['a','b','c'],2:['c','d','e']},
            1:{0:['i','j','k'],2:['f','g','h']}}
    
    output:
    {0: {1: ['i', 'j', 'k']},
     1: {0: ['a', 'b', 'c']},
     2: {0: ['c', 'd', 'e'], 1: ['f', 'g', 'h']}}
     
     Pseudocode: 
     How to flip the soma to piece touching dictionaries
    1) get all the possible limb keys
    2) Create a dictionary with empty list
    3) Iterate through all of the somas
    - if the limb is in the keys then add the info (if not then skip)

    
    """
    test_dict = curr_dict
    all_limbs = np.unique(np.concatenate([list(v.keys()) for v in test_dict.values()]))
    flipped_dict = dict()
    for l_idx in all_limbs:
        flipped_dict[l_idx]=dict()
        for sm_idx,sm_dict in test_dict.items():
            if l_idx in sm_dict.keys():
                flipped_dict[l_idx].update({sm_idx:sm_dict[l_idx]})
    return flipped_dict

#import itertools
def combine_list_of_lists(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


#have to reorder the keys
def order_dict_by_keys(current_dict):
    current_dict_new = dict([(k,current_dict[k]) for k in np.sort(list(current_dict.keys()))])
    return current_dict_new


def dict_to_array(current_dict):
    return np.vstack([list(current_dict.keys()),list(current_dict.values())]).T



def add_prefix_to_keys(data,prefix):
    return {f"{prefix}_{k}":v for k,v in data.items()}

#from datasci_tools import data_struct_utils as dsu
def merge_dicts(dicts):
    """
    Purpose: To combine multiple dictionaries
    
    Ex: 
    x = dict(a=5,b=8,c=9)
    y = dict(d = 10,e=7,f=10)
    z = dict(r = 20,g = 30)
    merge_dicts([x,y,z])
    """
    
    
    if len(dicts) == 0:
        return {}
    
    dicts = nu.to_list(dicts)
    
    if np.any([isinstance(k,dsu.DictType) for k in dicts]):
        dicts = [dsu.DictType(k) for k in dicts]
    
    super_dict = dicts[0].copy()
    for d in dicts[1:]:
        super_dict.update(d)
    return super_dict

def flatten_nested_dict(
    data,
    level_prefix = False,
    prefix = None,
    ):
    """
    Example: 
    ex_dict = dict(
        x = 10,
        my_dict = dict(another_dict = dict(yyy = 1000),y = 60,z = 100),
        zz = 70
    )

    gu.flatten_nested_dict(ex_dict,level_prefix = False)
    """
    if prefix is None:
        prefix = ""
    else:
        prefix = f"{prefix}_"
    new_dict = dict()
    other_dicts = []
    
    for k,v in data.items():
        if isinstance(v,dict):
            if level_prefix:
                send_prefix = f"{prefix}{k}"
            else:
                send_prefix = None
            other_dicts.append(
                flatten_nested_dict(
                    v,
                    level_prefix = level_prefix,
                    prefix = send_prefix,
                    ))
        else:
            new_dict[f"{prefix}{k}"] = v
            
    other_dicts.append(new_dict)
    return merge_dicts(other_dicts)

def print_nested_dict(
    d,
    indent=1,
    indent_incr=1,
    indent_str ="  ",
    total_str = "",
    filepath = None,
    overwrite = True,
    ):
    """
    Things still need to do: 
    1) Comment out all functions, list of functions, etc.
    2) Add np. in front of arrays
    """
    def file_export(filepath):
        nonlocal overwrite
        if filepath is None:
            return None
        else:
            if overwrite:
                mode = "w"
                overwrite = False
            else:
                mode = "a"
            return open(filepath,mode)
        
        
    print(f'{indent_str}' * (indent - 1) + "{",file=file_export(filepath))
    for key, value in d.items():
        if isinstance(value,dsu.DictType):
            value = value.asdict()
        if isinstance(value, dict):
            print(f'{indent_str}' * indent + f"{repr(key)}:",
                  file=file_export(filepath))
            print_nested_dict(
                value, 
                indent+indent_incr,
                indent_incr=indent_incr,
                indent_str=indent_str,
                filepath = filepath,
                overwrite=overwrite
            )
        else:
            print(f'{indent_str}' * (indent) + f"{repr(key)}:{repr(value)},",
                  file=file_export(filepath))
    end_str = f'{indent_str}' * (indent - 1) + "}"
    if indent-1 == 0:
        print(end_str,file=file_export(filepath))
    else:
        print(end_str + ",",file=file_export(filepath))
    

import trimesh
def nested_dict_obj_search(
    data_struct,
    class_to_find,
    objs_list = None,
    debug = False,
    ):
    """
    Purpose: To search a nested dictionary
    for certain object type, collect it,
    and then return a complete collections
    of those objects
    """
    
    
    if objs_list is None:
        objs_list= []
        
        
    return_value= None
    if debug:
        print(f"Working on {data_struct} (id = {id(data_struct)})")
        
    if isinstance(data_struct,class_to_find):
        if debug:
            print(f"Found instance of class: ")
        objs_list.append(data_struct)
        return_value= objs_list 
    else:
        #if ("dict" in str(data_struct.__class__).lower() or
        #  "list" in str(data_struct.__class__).lower() ):
        if "__iter__" in dir(data_struct) \
            and "str" not in str(type(data_struct)):
            if debug:
                print(f"is iterable")
            for k in data_struct:
                if "keys" in dir(data_struct):
                    value = data_struct[k]
                else:
                    value = k
                    
                if debug:
                    print(f"Going to search value {k} with value {value}")
                    
                return_value = nested_dict_obj_search(
                    value,
                    class_to_find = class_to_find,
                    objs_list = objs_list,
                    debug = debug
                )
        else:
            if debug:
                print("not iterable so returning")
            return_value = objs_list
            
    if debug:
        print(f"-- current objs_list --")
        for k in objs_list:
            print(f"{k} (id = {id(k)})")
        print(f"\n")
            
    return return_value

def remove_dict_suffixes(
    data,
    suffixes,
    ):
    """
    Purpose: To remove any suffixes from a diction
    """
    suffixes = nu.to_list(suffixes)
    new_data = dict()
    for k,v in data.items():
        new_name = k
        for suf in suffixes:
            if k[-len(suf):] == suf:
                new_name = k[:-len(suf)]
                break
        new_data[new_name] = v
    return new_data

#import itertools
def merge_dicts_simple(dicts):
    return dict(itertools.chain(*[k.items() for k in dicts]))


def is_function(obj):
    """
    Ex: 
    from datasci_tools import general_utils as gu

    def print_hello():
        print("hi")
    gu.is_function(print_hello)
    """
    return callable(obj)

#from . import numpy_dep as np
def sub_dict(obj,keys_to_include=None,
            keys_to_exclude=None):
    """
    Purpose: To restrict a dictionary
    
    Ex: 
    gu.sub_dict(dict(hello=5,hi=7),keys_to_exclude="hello")
    """
    curr_keys = list(obj.keys())
    
    if keys_to_include is not None:
        if not nu.is_array_like(keys_to_include):
            keys_to_include = [keys_to_include]
        curr_keys = np.intersect1d(curr_keys,keys_to_include)
    
    if keys_to_exclude is not None:
        if not nu.is_array_like(keys_to_exclude):
            keys_to_exclude = [keys_to_exclude]
        curr_keys = np.setdiff1d(curr_keys,keys_to_exclude)
        
    return {k:obj[k] for k in curr_keys}


# ------------ Help  with raising errors --------------- #
class Error(Exception): 
  
    # Error is derived class for Exception, but 
    # Base class for exceptions in this module 
    pass
  
class CGAL_skel_error(Error): 
  
    # Raised when an operation attempts a state  
    # transition that's not allowed. 
    def __init__(self,  msg): 
  
        # Error message thrown is saved in msg 
        self.msg = msg 
        
        
def str_filter_away_character(string, 
                            character_to_remove):
    """
    Ex: 
    s = "my_new_name"
    gu.str_filter_away_character(s,"_")
    
    Output: >> mynewname
    """
    return string.replace(character_to_remove,"")

def str_filter_away_characters(string,
                              characters_to_remove):
    for c in characters_to_remove:
        string = string.replace(c,"")
        
    return string

def add_prefix_to_dict_keys(data,prefix):
    return {f"{prefix}_{k}":v for k,v in data.items()}

        

        

        
        
        




#--- from datasci_tools ---
from . import data_struct_utils as dsu
from . import numpy_utils as nu
