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


import re
from pathlib import Path

def file_regex_replace(
    pattern,
    replacement,
    filepath,
    overwrite_file = False,
    output_filepath = None,# "text_revised.txt",
    default_suffix = "_replaced",
    verbose = False,
    regex = True,
    **kwargs
    ):
    """
    Purpose: To replace certain text 
    in a file and either write changes
    back to same file or to a new one

    Pseudocode:
    1) open the file
    2) read in the contents of the file
    3) using the regex replace function (with potential to use captured groups)
    """
    if overwrite_file:
        output_filepath = filepath
    
    if output_filepath is None:
        p = Path(filepath)
        output_filepath = f"{p.with_suffix('')}{default_suffix}{p.suffix}"


    # Opening our text file in read only
    # mode using the open() function
    data = filu.read_file(filepath)
    # Searching and replacing the text
    # using the replace() function
    if regex:
        data,count = re.subn(pattern, replacement, data)
        if verbose:
            print(f"# of substitutions = {count}")
    else:
        data = data.replace(pattern,replacement)

    # Opening our text file in write only
    # mode to write the replaced content
    filu.write_file(output_filepath,data)
        
    #return output_file
    
import io
def read_file(
    filepath,
    encoding="utf-8"
    ):
    d = None
    with io.open(filepath, "r", encoding=encoding) as my_file:
        d = my_file.read()
        
    return d

def write_file(filepath,data,encoding="utf-8"):
    with io.open(filepath, "w", encoding=encoding) as my_file:
        my_file.write(data)
    
    
    
def file_regex_add_prefix(
    pattern,
    prefix,
    filepath,
    overwrite_file = False,
    output_filepath = None,# "text_revised.txt",
    default_suffix = "_replaced",
    verbose = False,
    regex = True,
    **kwargs,
    ):
    """
    Purpose: Add a prefix in front of certain patterns in file
    
    Ex:
    file_regex_replace(
        pattern = "(new hello)",
        prefix = "from . ",
        filepath = "test.txt",
        overwrite_file=False,
        verbose = True
    )
    """

    return filu.file_regex_replace(
        pattern = pattern,
        replacement = fr"{prefix}" + r"\1",
        filepath=filepath,
        overwrite_file = overwrite_file,
        output_filepath = output_filepath,# "text_revised.txt",
        default_suffix = default_suffix,
        verbose = verbose,
        regex = regex,
        **kwargs
    )
    
    
import file_utils as filu