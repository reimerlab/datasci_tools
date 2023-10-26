'''



Utility functions for manipulating strings




'''
import copy
#from datasci_tools import hash_utils as hashu
#from datasci_tools import string_utils as stru
#import copy
#from datasci_tools import data_struct_utils as dsu

def abbreviate_str_keys(
    d,
    max_phrase_len = 20,
    error_on_non_str_dict = True,
    #for hashing the string
    hash_key = False,
    
    #for abbreviating str without hashing
    separating_character = "_",
    max_word_len = 2,
    include_separating_character = True,
    
    verbose_key_change = False,
    verbose = False,
    return_key_mapping=False,
    collision_resolution = True,
    ):
    """
    Purpose: to abbreviate
    the key strings of a dictionary
    
    Ex: 
    x = dict(
    this_is_a_really_long_name_and_unneccessary_hi = 5
    )

    from datasci_tools import dict_utils as dictu
    dictu.abbreviate_str_keys(
        x,
        verbose = True,
        max_word_len=3,
        max_phrase_len = 15
    )
    """

    total_key_length_before = 0
    total_key_length_after = 0
    
    if type(d) is dsu.DictType:
        my_dict = dsu.DictType()
        dict_type_flag = True
    else:
        my_dict = dict()
        dict_type_flag = False
        
    key_mapping = dict()
    
    for k,v in d.items():
        if type(k) != str and error_on_non_str_dict:
            raise Exception(f"{k} was not a string")
            
        if hash_key:
            new_key = hashu.hash_str(k,max_length=max_phrase_len)
        else:
            new_key = stru.abbreviate_phrase(
                k,
                max_phrase_len = max_phrase_len,
                separating_character = separating_character,
                max_word_len = max_word_len,
                include_separating_character = include_separating_character,)
            
        if verbose_key_change:
            print(f"{k} --> {new_key}")
            
        if new_key in key_mapping:
            if collision_resolution:
                new_key = k
            else:
                raise Exception(f"{k} and {key_mapping[new_key]} had conflict at {new_key}")
            
        key_mapping[new_key] = k
            
        total_key_length_before += len(k)
        total_key_length_after += len(new_key)
        
        if dict_type_flag:
            my_dict[new_key] = copy.copy((d._dict[k],d._types.get(k,None)))
        else:
            my_dict[new_key] = copy.copy(d[k])
        
    if verbose:
        print(f"\n\n")
        print(f"total_key_length_before = {total_key_length_before}")
        print(f"total_key_length_after = {total_key_length_after}")
        print(f"len dict before = {len(d)}, len dict after = {len(my_dict)}")
        
    if len(d) != len(my_dict):
        print(f"len dict before = {len(d)}, len dict after = {len(my_dict)}")
        raise Exception("There was a conflict")
        
    if return_key_mapping:
        return my_dict,key_mapping
    else:
        return my_dict







#--- from datasci_tools ---
from . import data_struct_utils as dsu
from . import hash_utils as hashu
from . import string_utils as stru
