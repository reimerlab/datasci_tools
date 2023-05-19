from . import numpy_utils as nu
from . import system_utils as su
from pathlib import Path
import tqdm
"""
utilizies the pipreqs tool to automatically
generate the requirements of a folder

"""


#from python_tools import system_utils as su
#from python_tools import numpy_utils as nu
#import tqdm
#from pathlib import Path
import numpy as np
from python_tools import file_utils as filu

def requirement_file_from_requirements_dict(
    requirement_dict,
    filepath = "./requirements.txt",
    mode = "gt"):
    
    if mode == "gt":
        comp = ">="
    else:
        comp = "=="
    data = "\n".join([f"{k}{comp}{v}," for k,v in requirement_dict.items()])

    filu.write_file(
        filepath = filepath,
        data = data
    )
    

def requirements_dict_from_directories(
    directories,
    verbose = False,
    files_to_ignore = None,# ("__init__.py",),
    output_to_package_directory = True,
    output_directory = "./",
    output_filename = "requirements.txt",
    diff_with_package_import_list = True,
    remove_egg_lines = True,
    delete_file = False,
    mode = "gt"
    ):
    """
    Purpose: To ouptut the requrimenet.txt files for a 
    list of directiories with modules inside and then
    to compile those requrimenets into a dictionary
    """
    packages = directories
    packages = nu.to_list(packages)
    
    pack_includes = dict()
    
    if files_to_ignore is None:
        files_to_ignore = []
    else:
        files_to_ignore = nu.to_list(files_to_ignore)
        
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(exist_ok = True)
        
    
    for k in tqdm.tqdm(packages):
        if not Path(k).exists():
            filepath = f"/{k}/{k}"
        else:
            filepath = k

        curr_files = list(Path(filepath).iterdir())
        ignore_files = [str(k) for k in curr_files if k.is_dir() or k.suffix != ".py"]
        ignore_files += [str((Path(filepath) / Path(k)).absolute()) for k in files_to_ignore]
        command = f"pipreqs --encoding=utf8 --force"
        command += f" {filepath}"
        if len(ignore_files) > 0:
            command += (f" --ignore=" + ",".join(ignore_files))
            
        if mode is not None:
            command += f" --mode {mode}"

        if len(packages) > 1:
            curr_filename = f"{output_filename}_{k}"
        else:
            curr_filename = output_filename
            
        if output_to_package_directory:
            #curr_output_directory = Path(filepath)
            curr_output_directory = Path(filepath).parents[0]
        else:
            curr_output_directory = output_directory
        output_path = str((curr_output_directory / f"{curr_filename}").absolute())
        
        command += f" --savepath {output_path}"
        
        if verbose:
            print(f"command = {command}")

        output = su.bash_command(command)
        requirement_dict = package_ver_dict_from_file(output_path)
        
        
        if diff_with_package_import_list:
            req_vs_package_import_diff(
                requirement_dict=requirement_dict,
                directory = filepath,
                verbose = True,        
            )
            
        Path(output_path).unlink()
            
            
        pack_includes[k] = requirement_dict
        
        if not delete_file:
            requirement_file_from_requirements_dict(
                requirement_dict=requirement_dict,
                filepath = output_path,
            )
        
    return pack_includes
from . import module_utils as modu
def req_vs_package_import_diff(
    requirement_dict,
    directory,
    verbose = False):
    req_mods = modu.package_imports_from_files(
        directory = directory,
    )
    
    import_diff = np.setdiff1d(
                req_mods,
                list(requirement_dict.keys())
    )
    
    if verbose:
        print(f"Package Diff: {import_diff}")
        
    return import_diff
    

def package_ver_dict_from_file(filepath):
    requirement_list = nu.array_from_txt(filepath,dtype = "str")
    try:
        requirement_dict = {pack:ver for pack,ver in [curr_split.split("==") for curr_split in requirement_list] if ".egg" != pack[-4:]}
    except:
        requirement_dict = {pack:ver for pack,ver in [curr_split.split(">=") for curr_split in requirement_list] if ".egg" != pack[-4:]}
    return requirement_dict

from . import requirement_utils as requ