'''



Utils functions for helping to work with files




'''
from pathlib import Path
import io
from . import numpy_dep as np
import os
import re

#from pathlib import Path

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


#import os
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


#import re
#from pathlib import Path

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
        
    #print(f"output_filepath = {output_filepath}")
    
    if output_filepath is None:
        p = Path(filepath)
        output_filepath = f"{p.with_suffix('')}{default_suffix}{p.suffix}"
        
    #print(f"output_filepath = {output_filepath}")


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
        
    if verbose:
        print(f"  --> output_filepath = {output_filepath}")

    # Opening our text file in write only
    # mode to write the replaced content
    filu.write_file(output_filepath,data)
        
    return output_filepath
    
#import io
def read_file(
    filepath,
    encoding="utf-8"
    ):
    d = None
    with io.open(filepath, "r", encoding=encoding) as my_file:
        d = my_file.read()
        
    return d

def write_file(filepath,data,encoding="utf-8",replace = False):
    with io.open(filepath, "w", encoding=encoding) as my_file:
        if replace:
            my_file.truncate(0)
        my_file.write(data)
    
    
    
def file_regex_add_prefix(
    pattern,
    prefix,
    filepath,
    replacement= None,
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
    if replacement is None:
        replacement = fr"{prefix}" + r"\1",

    return filu.file_regex_replace(
        pattern = pattern,
        replacement = replacement,
        filepath=filepath,
        overwrite_file = overwrite_file,
        output_filepath = output_filepath,# "text_revised.txt",
        default_suffix = default_suffix,
        verbose = verbose,
        regex = regex,
        **kwargs
    )
    
def file_regex_search(
    filepath,
    patterns,
    verbose = False,
    unique = False,
    ):
    """
    Purpose: To search a file for a certain regex pattern
    and to return all instances within the file

    Pseudocode: 
    1) Read in the file into a string
    2) Search the string for the patterns
    
    Example: 
    filu.file_regex_search(
        filepath = '/neurd_packages/mesh_tools/mesh_tools/skeleton_utils.py',
        patterns = [
            "nviz.([a-zA-Z1-9_-]*)\(",
            #"sm.([a-zA-Z1-9_-]*)\("
        ],
        verbose = True,
        unique = True,
    )
    """
    
    patterns = nu.to_list(patterns)
    patterns = f"(?:" + "|".join(patterns) + ")"

    if verbose:
        print(f"regex pattern = \n{patterns}")
        
    all_matches = []
    filepath = nu.to_list(filepath)

    for f in filepath:
        #1) Read in the file into a string
        data = filu.read_file(f)


        matches = reu.match_pattern_in_str(
            data,
            patterns 
        )

        if unique:
            matches = list(np.unique(matches))
        if verbose:
            print(f"# of matches in {Path(f).stem} = {len(matches)}")
            
        all_matches += matches
    
    all_matches = list(np.unique(all_matches))
    if verbose:
        print(f"# of matches total = {len(all_matches)}") 
        

    return all_matches

#--- from datasci_tools ---
from . import numpy_utils as nu
from . import regex_utils as reu


from . import file_utils as filu