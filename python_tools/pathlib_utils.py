
from pathlib import Path
from tqdm.notebook import tqdm
import os
import re
import regex as re
import shutil

def filename(path):
    return Path(path).name
def filename_no_ext(path):
    return Path(path).stem
def ext(path):
    return Path(path).suffix

def resolve(path):
    return Path(path).resolve()
relative_to_absolute_path = resolve

def parents(path):
    path = Path(path).resolve()
    return list(path.parents)


"""
How to search and unlink files: 

#import os
#import re

#from tqdm.notebook import tqdm

rootdir = Path("/mnt/dj-stor01/platinum/minnie65/02/decomposition/")
pattern = 'pipe_v7'
total_files = []

for f in tqdm(rootdir.iterdir()):
    if pattern in str(f.stem):
        #print(file)
        total_files.append(f)
        
for f in total_files:
    f.unlink()

"""

def create_folder(folder_path):
    """
    To create a new folder
    
    from python_tools from . import pathlib_utils as plu
    plu.create_folder("/mnt/dj-stor01/platinum/minnie65/02/graphs")
    """
    p = Path(folder_path)
    p.mkdir(parents=True, exist_ok=True)
    
#import shutil
def copy_file(filepath,destination):
    shutil.copy(str(filepath), str(destination))
    
def files_of_ext_type(
    directory,
    ext,
    verbose = False,
    return_stem = False,
    ):
    """
    Purpose: Get all files with a certain extension
    """
    if ext[0] == ".":
        ext = ext[1:]
    files = [k for k in Path(directory).iterdir() if k.suffix == f".{ext}"]
    if verbose:
        print(f"# of {ext} files = {len(files)}")
        
    if return_stem:
        files = [k.stem for k in files]
    return files

def py_files(
    directory,ext = "py",verbose = False,return_stem = False,):
    return files_of_ext_type(
    directory,
    ext=ext,
    verbose = verbose,
    return_stem=return_stem,
    )

def inside_directory(directory,filepath):
    """
    Ex: 
    from pathlib import Path
    from python_tools from . import pathlib_utils as plu

    root = Path("/python_tools/python_tools/")#.resolve()
    child = Path("../python_tools/numpy_utils.py")#.resolve()
    plu.inside_directory(root,child)
    """
    return Path(directory).resolve() in parents(filepath)

    
def relative_path_of_parent(parent,filepath):
    """
    Purpose: Find the relative path to parent
    """
    return str(Path(filepath).relative_to(Path(parent))).replace("//","/")

def n_levels_parent_above(parent,filepath,verbose = False):
    """
    Purpose: Find the number of levels a parent
    directory is above a filepath

    Pseudocode:
    1) get the relative path
    2) count the number of backslashes
    
    Ex: 
    plu.n_levels_parent_above(
        filepath = Path("/python_tools/python_tools/dj_utils.py"),
        parent = "/python_tools/",
        verbose = True
    )
    """
    relative_path = plu.relative_path_of_parent(parent,filepath)
    n_levels = len(re.findall(re.compile("/"),relative_path))
    if verbose:
        print(f"n_levels above = {n_levels}")
    return n_levels


#from python_tools from . import pathlib_utils as plu


from . import pathlib_utils as plu