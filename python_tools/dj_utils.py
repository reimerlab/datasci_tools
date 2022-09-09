"""
Purpose: Datajoint utils to help with table manipulation
"""
import datajoint as dj
import numpy_utils as nu
import numpy as np
import pandas as pd
import pandas_utils as pu

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
        features = nu.convert_to_array_like(features)
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

def df_from_table(
    table,
    features=None,
    remove_method_features = False,
    features_to_remove = None,
    features_substr_to_remove = None):
    
    all_atts = list(all_attribute_names_from_table(table))
    
    if features is None:
        features = all_atts
    
    features = nu.convert_to_array_like(features)
    
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

    if source_name is not None and len(source_name) > 0:
        source_name = f"{source_name}_"

    if target_name is not None and len(target_name) > 0:
        target_name = f"{target_name}_"

    proj_table = table_to_append.proj(*all_attributes)
    final_table = table

    rename_dict = {}
    for name,k in zip(["source","target"],[source_name,target_name]):
        if append_type == "prefix":
            name_str = 'f"{kk}{v}"'
        elif append_type == "suffix":
            name_str = 'f"{v}{kk}"'
        elif append_type is None:
            name_str = 'f"{kk}"'
        else:
            raise Exception("")

        rename_dict.update(dict([(eval(name_str),v) if v not in attributes_to_not_rename
                                 else (v,v) for kk,v in zip([k]*len(all_attributes),all_attributes)]))

        if verbose:
            print(f"rename_dict for {name} = {rename_dict}")

        final_table = final_table * proj_table.proj(**rename_dict) 

    return final_table




restrict_table_from_list = pu.restrict_df_from_list

import dj_utils as dju