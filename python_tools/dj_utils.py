'''

Purpose: Datajoint utils to help with table manipulation

'''
import datajoint as dj
import numpy as np
import pandas as pd
import time
#import datajoint as dj
#from python_tools from . import numpy_utils as nu
#import numpy as np
#import pandas as pd
#from python_tools from . import pandas_utils as pu

def df_from_table_old(
    table,
    features=None,
    remove_method_features = False,):
    
    if remove_method_features:
        if features is None:
            features = [k for k in all_attribute_names_from_table(table)
                    if 'method' not  in k]
        else:
            features = [k for k in features if "method" not in k]
    
    if features is not None:
        features = nu.convert_to_array_like(features,include_tuple = True)
        #table = table.proj(*features)
        #table = dj.U(*features) & table
        if len(features) == 1:
            curr_data = np.array(table.fetch(*features)).reshape(-1,len(features))
        else:
            curr_data = np.array(table.fetch(*features)).T
        return_df = pd.DataFrame(curr_data)
        return_df.columns = features
    else:
        return_df = pd.DataFrame(table.fetch())
        
    #return_df = pd.DataFrame.from_records(table.fetch())
#     if features is not None:
#         if isinstance(features,tuple):
#             features = list(features)
#         return_df = return_df[features]
    return return_df

#import time
def df_from_table(
    table,
    features=None,
    remove_method_features = False,
    features_to_remove = None,
    features_substr_to_remove = None,
    primary_features = False,
    verbose = False):
    
    st = time.time()
    
    all_atts = list(all_attribute_names_from_table(table))
    
    if features is None:
        features = all_atts
    
    features = nu.convert_to_array_like(features,include_tuple = True)
    
    if primary_features:
        primary_features = list(primary_attribute_names_from_table(table))
        features =primary_features + [k for k in features if k not in primary_features]
    
    if features_to_remove is None:
        features_to_remove = []
        
    if remove_method_features:
        features_to_remove += [k for k in all_atts if "method" in k]
        
    if features_substr_to_remove is not None:
        for att in all_atts:
            remove_flag = False
            for sub in features_substr_to_remove:
                if sub in att:
                    remove_flag = True
                    break
            if remove_flag:
                features_to_remove.append(att)
            
    features = [k for k in features if k not in features_to_remove]
    
    if len(features) == 1:
        curr_data = np.array(table.fetch(*features)).reshape(-1,len(features))
    else:
        curr_data = np.array(table.fetch(*features)).T
    return_df = pd.DataFrame(curr_data)
    return_df.columns = features
    
    if verbose:
        print(f"Time for fetch = {time.time() - st}")
    
    return return_df

def all_attribute_names_from_table(table):
    return [str(k) for k in table.heading.names]

def primary_attribute_names_from_table(table):
    return [str(k) for k in table.heading.primary_key]

def secondary_attribute_names_from_table(table):
    return [str(k) for k in table.heading.secondary_attributes]

def append_table_to_pairwise_table(
    table,
    table_to_append,
    primary_attributes = None,
    secondary_attributes = None,
    attributes_to_not_rename = None,
    source_name = "",
    target_name = "match",
    append_type = "prefix",
    verbose = False,
    ):
    """
    Purpose: To add on a table to both 
    sides of another

    Pseudocode: 
    1) Determine the primary attributes to join on
    (if not already)
    2) Determine the secondary attributes you want to 
    join
    3) Build a renaming dictionary
    4) Project the tale to only those attributes chosen
    5) star restrict the table with the renamed tables
    
    --- Example: ----
    dju.append_table_to_pairwise_table(
        table = m65auto.AutoProofreadNeuronLeafMatch(),
        table_to_append = hdju.subgraph_vectors_table,
        primary_attributes = [
            "segment_id",
            "split_index",
            "subgraph_idx",
            #"leaf_node",
        ],
        secondary_attributes = [
         'compartment',
         'node',
         'y_soma_relative',
         'width',
        ],
        source_name = "",
        target_name = "match",
        append_type = "prefix",
        verbose = True,
    )
    """
    if attributes_to_not_rename is None:
        attributes_to_not_rename = []

    if primary_attributes is None:
        primary_attributes = primary_attribute_names_from_table(table_to_append)

    if secondary_attributes is None:
        secondary_attributes = secondary_attribute_names_from_table(table_to_append)

    all_attributes = list(primary_attributes) + list(secondary_attributes)

#     if source_name is not None and len(source_name) > 0:
#         source_name = f"{source_name}_"

#     if target_name is not None and len(target_name) > 0:
#         target_name = f"{target_name}_"

    proj_table = table_to_append.proj(*all_attributes)
    final_table = table

    
    for name,k in zip(["source","target"],[source_name,target_name]):
        rename_dict = {}
        name_str = None
        if k is not None and len(k) > 0:
            if append_type == "prefix":
                name_str = 'f"{kk}_{v}"'
            elif append_type == "suffix":
                name_str = 'f"{v}_{kk}"'
            elif append_type is None:
                name_str = 'f"{kk}"'
        
        
        if name_str is None:
            name_str = 'f"{v}"'
            #raise Exception("")
            
        
        #print(f"name_str = {name_str}")
        rename_dict.update(dict([(eval(name_str),v) if v not in attributes_to_not_rename
                                 else (v,v) for kk,v in zip([k]*len(all_attributes),all_attributes)]))

        if verbose:
            print(f"rename_dict for {name} = {rename_dict}")

        final_table = final_table * proj_table.proj(**rename_dict) 

    return final_table


parameter_datatype_lists = [
    "int unsigned",
    "float",
    "double",
    "tinyint unsigned"
]
def parameter_datatype(
    parameter,
    int_default = "int",
    default_type = None,
    blob_type = "longblob"):
    
    if default_type in parameter_datatype_lists:
        return default_type
    
    curr_type_str = str(type(parameter))
    if default_type is not None:
        curr_type_str = str(default_type)
    
    if "int" in curr_type_str:
        try:
            if parameter >= 0:
                return f"{int_default} unsigned"
            else:
                return f"{int_default}"
        except:
            return f"{int_default}"
    elif "str" in curr_type_str:
        return f"varchar({len(parameter)+ 10})"
    elif ("float" in curr_type_str) or ("double" in curr_type_str):
        return "float"
    elif "bool" in curr_type_str:
        return "tinyint unsigned"
    elif "tuple" or "blob" in curr_type_str:
        return blob_type
    else:
        raise Exception(f"Unknown type: {type(parameter)}")
        
#from python_tools from . import data_struct_utils as dsu
def parameter_datatype_description(kwargs_dict,
                                  kwargs_datatype_dict = None,
                                   add_null = True,
                                  verbose = False,):
    """
    To generate the the datatype part of a parameter 
    table definition
    
    --- Exmaple--
    new_str = parameter_datatype_description(kwargs_dict,
                                         kwargs_datatype_dict = dict(filter_by_volume_threshold="new datatype")
                                        )
    print(new_str)
                                        
    """
    if kwargs_datatype_dict is None:
        kwargs_datatype_dict = {}
    total_str = []
    for k,v in kwargs_dict.items():
        if k in kwargs_datatype_dict:
            datatype_str = kwargs_datatype_dict[k]
        else:
            if isinstance(kwargs_dict,dsu.DictType):
                default_type = kwargs_dict._types.get(k,None)
            else:
                default_type = None
            datatype_str = parameter_datatype(v,default_type=default_type)
        
        if add_null:
            curr_str= f"{k}=NULL: {datatype_str}"
        else:
            curr_str= f"{k}: {datatype_str}"
            
        if verbose:
            print(curr_str)
        total_str.append(curr_str)
        
    return "\n".join(total_str)

def table_definition_from_example_dict(
    kwargs_dict,
    kwargs_datatype_dict = None,
    definition_description = None,
    add_name_description_to_parameters = False,
    verbose = False
    ):
    
    total_str = []
    if definition_description is not None:
        if definition_description[0] != "#":
            total_str.append(f"# {definition_description}")
        else:
            total_str.append(definition_description)
            
    total_str.append("->master\n---")
    param_str = parameter_datatype_description(kwargs_dict,
                                              kwargs_datatype_dict = kwargs_datatype_dict,
                                              verbose = False)
    total_str.append(param_str)
    
    if add_name_description_to_parameters:
        if "name" not in kwargs_dict:
            total_str.append("name=NULL: varchar(24)")
        if "description" not in kwargs_dict:
            total_str.append("description=NULL: varchar(120)")
        
    total_str_comb = "\n".join(total_str)
    
    if verbose:
        print(total_str_comb)
        
    return total_str_comb

def dj_query_from_pandas_query(query):
    query = query.replace("==","=")
    for joiner in [" and "," or "," not "]:
        query = query.replace(joiner,joiner.upper())
    return query

def query_table_from_kwargs(
    table,
    **kwargs):
    """
    Purpose: To query a datajoint table
    with keywords that may be None
    """
    key = dict()
    key = {k:v for k,v in kwargs.items() if v is not None}
    if len(key) == 0:
        return table
    else:
        return table & key

restrict_table_from_list = pu.restrict_df_from_list

#from python_tools from . import dj_utils as dju

#--- from python_tools ---
from . import data_struct_utils as dsu
from . import numpy_utils as nu
from . import pandas_utils as pu

from . import dj_utils as dju