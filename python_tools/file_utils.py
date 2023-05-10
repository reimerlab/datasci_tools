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

def import_pattern_str(
    start = None,
    beginning_of_line = True):
    
    word_comb = "[a-zA-Z._]+"
    
    if start is None:
        if beginning_of_line:
            start = "\n"
        else:
            start = ""
    return (f"{start}("
        f"(?:import {word_comb} as {word_comb})"
        f"|(?:from {word_comb} import {word_comb} as {word_comb})"
        f"|(?:import {word_comb})"
        f"|(?:from {word_comb} import {word_comb})"          
    ")"
    )

def find_import_modules_in_file(
    filename,
    unique = True,
    verbose = False,
    beginning_of_line = True,
    ):
    """
    Purpose: Find all imports (optionally outside of functions)
    in a file and create a list (most likely unique)

    Pseudocode: 
    1) read in file data
    2) create the pattern to recognize imports
    3) add the newline if only care about newline
    4) Search string for all matches
    5) Create a list (maybe unique)
    """


    #1) read in file data
    data = filu.read_file(filename)

    #2) create the pattern to recognize imports
    

    

    pattern = filu.import_pattern_str(beginning_of_line=beginning_of_line)

    re.compile(pattern)
    finds = list(re.finditer(pattern,string=data))

    str_finds = [f.string[f.start():f.end()].replace('\n',"") for f in finds]

    if unique:
        str_finds = list(set(str_finds))

    if verbose:
        print(f"# of matches (unique = {unique}) = {len(str_finds)}")

    return str_finds

from pathlib import Path
import numpy as np
def clean_module_imports(
    filename,
    overwrite_file = False,
    verbose = False,
    relative_package = "python_tools",
    relative_replacement = ".",
    ):
    """
    Want to get all of the modules importing in file
    (could then make a unique list and change) certain 
    ones to newline

    -- then put at the top

    Pseudocode: 
    1) find all statements where import is at the beginning of the line
    -> either from [] or import
    2) Extract the modules and the from for all
    3) Create a unique list
    4) Replace any requested modules in from [] with another prefix
    5) That is list need to copy and paste at top
    6) Comment all of the rest out
    """

    curr_file = Path(filename).stem

    #1-3) Create a unique list
    finds = filu.find_import_modules_in_file(
        filename  =filename,
        unique = True,
        verbose = verbose,
        beginning_of_line = True,
    )

    finds_top = [k for k in finds if curr_file not in k]
    ending_import = [k for k in finds if curr_file in k]

    #6) Comment all of the rest out
    output_file = filu.file_regex_replace(
        filepath=filename,
        pattern = filu.import_pattern_str(beginning_of_line = True),
        replacement = "\n#" + r"\1",
        overwrite_file = overwrite_file,
        verbose = verbose
    )

    if relative_package is not None:
        #4) Replace any requested modules in from [] with another prefix
        finds_top = [k.replace(relative_package,relative_replacement) if "." not in k else
                     k.replace(relative_package,"") for k in finds_top]
        
        ending_import = [k.replace(relative_package,relative_replacement) for k in ending_import]

    finds_top = list(np.sort(finds_top))
    #5) That is list need to copy and paste at top
    finds_top_str = "\n".join(finds_top)
    ending_import = f"\n\n{''.join(ending_import)}"

    data = filu.read_file(output_file)
    filu.write_file(filepath=output_file,data=finds_top_str + f"\n" + data + ending_import)
    return output_file

    
    
from python_tools import file_utils as filu