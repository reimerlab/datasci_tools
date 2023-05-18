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


def requirements_dict_from_directories(
    directories,
    verbose = False,
    files_to_ignore = None,# ("__init__.py",),
    output_to_package_directory = True,
    output_directory = "./",
    output_filename = "requirements.txt",
    delete_file = False,
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
        requirement_list = nu.array_from_txt(output_path,dtype = "str")
        requirement_dict = {pack:ver for pack,ver in [curr_split.split("==") for curr_split in requirement_list] if ".egg" != pack[-4:]}
        pack_includes[k] = requirement_dict
        
        if delete_file:
            Path(output_path).unlink()
        
    return pack_includes


from . import requirement_utils as requ