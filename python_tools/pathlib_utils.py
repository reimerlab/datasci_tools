from pathlib import Path

def filename(path):
    return Path(path).name
def filename_no_ext(path):
    return Path(path).stem
def ext(path):
    return Path(path).suffix


"""
How to search and unlink files: 

import os
import re

from tqdm.notebook import tqdm

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
    
    import pathlib_utils as plu
    plu.create_folder("/mnt/dj-stor01/platinum/minnie65/02/graphs")
    """
    p = Path(folder_path)
    p.mkdir(parents=True, exist_ok=True)
    
import shutil
def copy_file(filepath,destination):
    shutil.copy(str(filepath), str(destination))
    
def files_of_ext_type(
    directory,
    ext,
    verbose = False,
    ):
    """
    Purpose: Get all files with a certain extension
    """
    files = [k for k in Path(directory).iterdir() if k.suffix == f".{ext}"]
    if verbose:
        print(f"# of {ext} files = {len(files)}")
    return files
    


import pathlib_utils as plu