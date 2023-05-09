"""
utilizies the pipreqs tool to automatically
generate the requirements of a folder

"""


from python_tools import system_utils as su
from python_tools import numpy_utils as nu
import tqdm
from pathlib import Path


def requirements_dict_from_directories(
    directories,
    verbose = False,
    files_to_ignore = None,
    output_filepath = "./requirements_output",
    ):
    """
    Purpose: To ouptut the requrimenet.txt files for a 
    list of directiories with modules inside and then
    to compile those requrimenets into a dictionary
    """
    packages = directories
    
    pack_includes = dict()
    
    if files_to_ignore is None:
        files_to_ignore = []
        
    if output_filepath is not None:
        output_filepath = Path(output_filepath)
        output_filepath.mkdir(exist_ok = True)
        
    for k in tqdm.tqdm(packages):
        if not Path(k).exists():
            filepath = f"/{k}/{k}"
        else:
            filepath = k

        curr_files = list(Path(filepath).iterdir())
        ignore_files = [str(k) for k in curr_files if k.is_dir() or k.suffix != ".py"]
        ignore_files += files_to_ignore
        command = f"pipreqs --encoding=utf8 --force"
        command += f" {filepath}"
        if len(ignore_files) > 0:
            command += (f" --ignore=" + ",".join(ignore_files))

        if output_filepath is not None:
            output_path = str((output_filepath / f"{k}_requirements.txt").absolute())
            command += f" --savepath {output_path}"
        if verbose:
            print(f"command = {command}")



        output = su.bash_command(command)
        requirement_list = nu.array_from_txt(f"{filepath}/requirements.txt",dtype = "str")
        requirement_dict = {pack:ver for pack,ver in [curr_split.split("==") for curr_split in requirement_list]}
        pack_includes[k] = requirement_dict
        
    return pack_includes