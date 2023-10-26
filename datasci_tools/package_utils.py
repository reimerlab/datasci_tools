'''



Notes: global variables can be referenced in functions
but can't be assigned to (if they are then its just a local copy)
without the use of the global keyword

Module importing itself: essentially will run through things twice
explanation: https://stackoverflow.com/questions/62665924/python-program-importing-itself

--> could technically put at the top




'''
from pathlib import Path
import io
import importlib
from os import sys
import os
#from datasci_tools import numpy_utils as nu
#from datasci_tools import file_utils as filu
#import io

user_packages = (
        "/neurd_packages/datasci_tools/datasci_tools/",
        "/neurd_packages/machine_learning_tools/machine_learning_tools/",
        "/neurd_packages/pytorch_tools/pytorch_tools/",
        "/neurd_packages/graph_tools/graph_tools/",
        "/neurd_packages/meshAfterParty/meshAfterParty/",
        "/neurd_packages/neuron_morphology_tools/neuron_morphology_tools/",
        "/neurd_packages/mesh_tools/mesh_tools/",
        "/neurd_packages/NEURD/neurd"
)

def module_names_from_directories(
    directories,
    ignore_files = ("__init__",),
    return_regex_or = False,
    verbose = False):
    """
    Purpose: come up with an or string for module names from a module directory
    """
    directories = nu.to_list(directories)
    modules = []

    for directory in directories:

        if verbose:
            print(f"--Getting files from {directory}")

        modules += plu.files_of_ext_type(
            directory = directory,
            ext = "py",
            verbose = verbose
        )
        
    modules = [k.stem for k in modules if k.stem not in ignore_files]
    if return_regex_or:
        return f"({'|'.join(modules)}))"
    else:
        return modules
    
def relative_import_from(directory,filepath):
    return "from " + "".join(["."]*(plu.n_levels_parent_above(directory,filepath)+1)) + " "

def prefix_module_imports_in_files(
    filepaths=None,
    modules_directory = "../datasci_tools",
    modules = None,
    filepaths_directory = None,
    prefix = "directory",
    auto_detect_relative_prefix = True,
    prevent_double_prefix = True,
    overwrite_file = False,
    output_filepath = None,# "text_revised.txt",
    verbose = False,
    ignore_files = ["__init__"],
    skip_filepath_parent_dir = True,
    debug_files = False,
    ):
    """
    want to add a prefixes before
    modules that are imported in a file

    Pseudocode: 
    1) if given a directory: get a list of the module names
    2) Construct a regex pattern ORing the potential list
    3) add the prefix before
    4) write to a new file or old file

    """
    if modules_directory is None:
        modules_directory = [None]
    
    modules_directory = [Path(k) for k in nu.to_list(modules_directory)]
    
    if filepaths_directory is not None:
        filepaths_directory = [Path(k) for k in nu.to_list(filepaths_directory)]
        filepaths = []
        for f_dir in filepaths_directory:
            if Path(f_dir) in modules_directory:
                if verbose:
                    print(f"skipping filepath directory {f_dir}")
                continue
            fpaths = plu.files_of_ext_type(
                directory = f_dir,
                ext = "py",
                verbose = False
            )
            filepaths += fpaths

    filepaths = nu.to_list(filepaths)
    
    for f in filepaths:
        if verbose or debug_files:
            print(f"--- Working on file: {f}")
            
        if output_filepath is not None:
            curr_output_filepath = output_filepath.copy()
        else:
            curr_output_filepath = output_filepath
            
            
        for directory in modules_directory:
        # --- iterate through all package directories and do the replacement

            #1) if given a directory: get a list of the module names
            if directory is not None:
                if verbose or debug_files:
                    print(f"--Getting files from {directory}")
                    
                
                    
                if plu.inside_directory(directory,f) and skip_filepath_parent_dir:
                    if verbose:
                        print(f"skipping {directory} for file {f}")
                    continue
                
                if debug_files:
                    continue

                modules = plu.files_of_ext_type(
                    directory = directory,
                    ext = "py",
                    verbose = verbose
                )

                if auto_detect_relative_prefix and plu.inside_directory(directory,f):
                    curr_prefix = relative_import_from(directory,f)
                elif prefix == "directory":
                    curr_prefix = f"from {Path(directory).stem} "
                else:
                    curr_prefix = prefix
            else:
                modules = [Path(k) for k in nu.to_list(modules)]
                curr_prefix = prefix

            modules = [k.stem for k in modules if k.stem not in ignore_files]

            #2) Construct a regex pattern ORing the potential list
            pattern = f"(import ({'|'.join(modules)}))"

            replacement = fr"{curr_prefix}import \2"
            if prevent_double_prefix:
                #pattern = f"(?<!{curr_prefix}){pattern}"
                pattern = f"(?<!{curr_prefix})(?<!from . ){pattern}"
            else:
                replacement = None

            #print(f"pattern = {pattern}")
            #print(f"replacement = {replacement}")

            #4) write to a new file or old file           
            curr_output_filepath = filu.file_regex_add_prefix(
                pattern=pattern,
                prefix=prefix,
                filepath=f,
                replacement=replacement,
                overwrite_file = overwrite_file,
                output_filepath = curr_output_filepath,# "text_revised.txt",
                verbose = verbose,
                regex = True,
            )
            
            f = curr_output_filepath

            
def package_name_from_path(path):
    return path.split("/")[-3]
def package_from_filepath_and_package_list(
    filepath,
    packages,
    return_package_name = False,
    verbose = False
    ):
    """
    packages = (
            "/datasci_tools/datasci_tools/",
            "/machine_learning_tools/machine_learning_tools/",
            "/pytorch_tools/pytorch_tools/",
            "/graph_tools/graph_tools/",
            "/meshAfterParty/meshAfterParty/",
            "/neuron_morphology_tools/neuron_morphology_tools/",
    )
    pku.package_from_filepath_and_package_list(
        filepath = "../datasci_tools/networkx_utils.py",
        packages=packages
    )
    
    """
    if verbose:
        print(f"Search for package for {filepath}")
    for p in packages:
        if plu.inside_directory(p,filepath):
            if verbose:
                print(f"Matched to {p}")
            if return_package_name:
                return package_name_from_path(p)
            return p
    return None

def create_init(
    directory,
    init_filename = "__init__.py",
    exist_ok = True):
    """
    Purpose: To create an init file in a directory
    if one doesn't already exist
    """
    filepath = Path(directory) / Path(init_filename)
    filepath.touch(exist_ok= exist_ok)
    return filepath

def clean_package_syntax(
    directory,
    overwrite=False,
    create_init_if_not_exist = True,
    verbose = False,
    packages = None,
    top_imports = None,
    ):
    """
    Purpose: To clean all modules in all
    of the given directories
    """

    directory = nu.to_list(directory)

    for curr_directory in directory:
        if verbose:
            print(f"---- working on directory: {curr_directory} ---")
        modules = modu.modules_from_directory(curr_directory)
        modules

        for mod in modules:
            filepath = Path(curr_directory) / (f"{mod}.py")
            if verbose:
                print(f"   --- working on module: {filepath} ---")
            modu.clean_module_syntax(
                filepath = filepath,
                verbose = verbose,
                overwrite=overwrite,
                packages=packages,
                top_imports=top_imports,
            )    
            
        if create_init_if_not_exist:
            pku.create_init(
                directory = curr_directory
            )
            

def module_files_from_directory(
    directory
    ):
    modules = pku.module_names_from_directories(directory)
    files = [directory / Path(f"{k}.py") for k in modules]
    return files

def modules_from_package_in_sys_modules(package_name):
    return  [k for k in sys.modules.values() if f"'{package_name}." in str(k)]

def submodule_version_dict(
    submodule,#= "hdju",
    package_name,
    use_loader = False,
    verbose = False,
    ):
    hdju_dict = dict()
    for mod in pku.modules_from_package_in_sys_modules(package_name):
        attr = getattr(mod,submodule,None)
        #print(f"mod = {mod}, attr = {attr}")
        if attr is not None:
            if use_loader:
                attr = getattr(attr,"__loader__")
            if str(attr) not in hdju_dict:
                hdju_dict[str(attr)] = []
                
            hdju_dict[str(attr)].append(mod)
            
    if verbose:
        for k,v in hdju_dict.items():
            print(f"{k}")
            for vv in v:
                new_name = vv.__name__.replace(f"{package_name}.","")
                print(f"   {new_name}")
                
        
    return hdju_dict

def reload_all_modules_in_package(package_name):
    """
    Purpose: To reload all modules in a specified package

    """
    for module in modules_from_package_in_sys_modules(package_name):
        importlib.reload(module)


def load_all_modules_in_package(
    package_directory,
    reload_after_load = True,
    suppress_import_errors = True,
    verbose=False,):
    
    if verbose:
        print(f"package_directory = {package_directory}")
    for module in os.listdir(os.path.dirname(__file__)):
        if module == '__init__.py' or module[-3:] != '.py':
            continue
        try:
            exec(f"from . import {module[:-3]}", locals(), globals())
        except Exception as e:
            if suppress_import_errors:
                pass
            else:
                raise Exception(e)
        
    if reload_after_load:
        reload_all_modules_in_package(Path(package_directory).stem)


#from datasci_tools import package_utils as pku




#--- from datasci_tools ---
from . import file_utils as filu
from . import module_utils as modu
from . import numpy_utils as nu
from . import pathlib_utils as plu

from . import package_utils as pku