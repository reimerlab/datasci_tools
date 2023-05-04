"""
Utils functions for helping to work with files

"""

from pathlib import Path

def file_from_dicts(
    dicts,
    filename=None,
    directory="./",
    filepath = None,
    seperation_character = ",",
    verbose = False,
    verbose_line = False,
    ):
    """
    Purpose: Convert list of dictionaries to 
    a file with one dictionary per line
    """
    if filepath is None:
        filepath = Path(directory) / Path(filename)
    else:
        filepath = Path(filepath)
        
    filepath = str(filepath.absolute())
        
    if verbose:
        print(f"Writing file to {filepath}")


    with open(filepath,"w") as f:
        for i,d in enumerate(dicts):
            line_to_write = seperation_character.join([str(k) for k in d.values()])+"\n"
            if verbose_line:
                print(f"line {i}= {line_to_write}")

            f.write(line_to_write)
            
    return filepath


import os
def str_in_filepath(
    search_str,
    filepath,
    dir_path = None,
    verbose = True,
    ):
    
    if dir_path is not None:
        filepath = os.path.join(dir_path, filepath)
    cur_path = filepath
    
    if os.path.isfile(cur_path):
        try:
            with open(cur_path, 'r') as file:
                # read all content of a file and search string
                if search_str in file.read():
                    if verbose:
                        print(f'{search_str} found in {cur_path}')
                    return True
        except:
            if verbose:
                print(f"    Wasn't able to read {cur_path}")
    return False
        
    
def search_directory_files_for_str(
    dir_path,
    search_str,
    verbose = True):
    
    all_files = []
    # iterate each file in a directory
    for file in os.listdir(dir_path):
        cur_path = os.path.join(dir_path, file)
        # check if it is a file
        if str_in_filepath(search_str,cur_path,verbose = verbose):
            all_files.append(cur_path)
            
    return all_files