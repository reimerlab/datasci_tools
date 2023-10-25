'''



Purpose: to help parse json files




'''
import json
import pandas as pd
from pathlib import Path
#from datasci_tools import pandas_utils as pu
#import pandas as pd
#import json
indent_default = 4

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
    
#from datasci_tools import json_utils as ju

def json_to_dict(filepath):

    with open(filepath) as json_file:
        data = json.load(json_file)

    return data

def dict_to_json(
    data,
    indent = None):
    
    if indent is None:
        indent = indent_default 
    # Serializing json  
    return json.dumps(data, indent = indent) 


def dict_to_json_file(
    data,
    filepath="./data.json",
    indent = None):
    
    filepath = str(Path(filepath).absolute())
    
    if indent is None:
        indent = indent_default 
    
    if filepath[-5:] != ".json":
        filepath += ".json"
    
    with open(filepath, "w") as outfile:
        json.dump(data, outfile,indent=indent)

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
    
def json_to_flat_dict(filepath,**kwargs):
    curr_json = jsu.json_to_dict(filepath,**kwargs)
    return gu.flatten_nested_dict(curr_json)
#--- from datasci_tools ---
from . import pandas_utils as pu
from . import general_utils as gu

from . import json_utils as ju