"""
Purpose: to help parse json files

"""
import pandas_utils as pu
import pandas as pd
import json

def df_from_json_file(filepath):
    return pu.json_to_df(filepath)

def flatten_df_from_json(df):
    pass
    

def dicts_list_from_json_file(filepath,flatten=True):
    """
    Convert a json file into a list of dictionaries
    """
    syn_df = pu.json_to_df(filepath)
    if not flatten:
        return pu.dataframe_to_row_dicts(syn_df)
    else:
        return pu.flatten_nested_dict_df(syn_df)
    
def json_from_file(filepath,multiple_objs = True):
    if multiple_objs:
        objs = []
        with open(filepath) as f:
            for jsonObj in f:
                studentDict = json.loads(jsonObj)
                objs.append(studentDict)
        return objs
    else:
        with open(filepath) as f:
            studentDict = json.loads(f)
        return studentDict
    
import json_utils as ju