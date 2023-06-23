"""
Purpose: to provide functionality for inspecting,
editing and interacting with python modules (the .py files)
"""
from . import module_utils as modu
from . import data_struct_utils as dsu


"""
Note: 

---Default arguments:---
Default arguments of a function only get initialized once, so if made
default arguments equal to a global value then would only take on the
default value of the global value that was there at the very beginning

Solution: Make equal to None and then set it equal to the global
value inside the funciton

---- using eval and exec ----
Rule: Can't set 


"""

"""
Demo: Shows that you can't set a local variable with exec and eval 

import inspect
x_global = 100
y_global = 200

def my_func(x = 2,y=None):
    inspect_elem = inspect.getargspec(my_func)
    for arg in inspect_elem.args:
        print(eval(arg))
        if eval(arg) is None:
            
            print(f"{arg} is None")
            command = f"{arg} = {arg}_global"
            print(f"{command}")
            exec(f"{command}")
            
    print(f"x = {x}, y = {y}")



"""
class options:
    def __init__(self, 
                **kwargs):
        self.options = list(kwargs.values())
        for k,v in kwargs.items():
            setattr(self,k,v)

    def __call__(self, f):
        f.directional = self.directional
        f.multiedge = self.multiedge
        return f



global_ending = "global"
output_types_global = ("global_parameters","attributes")


def basic_global_names(k,
                       global_suffix=None,
                      suffix_to_add = None):
    if global_suffix is None:
        global_suffix = global_ending
    if suffix_to_add is not None:
        global_suffix = f"{suffix_to_add}_{global_suffix}"
    if global_suffix == "":
        return k,k
        
    if k.split('_')[-1] == f"{global_suffix}":
        basic_name = '_'.join(k.split('_')[:-1])
        global_name = k
    else:
        basic_name = k
        global_name = f"{k}_{global_suffix}"
        
    return basic_name,global_name

def generate_None_assignment_block(param_dict,
                                  suffix_to_filter_away=None,
                                  suffix_to_add = None):
    for k in param_dict.keys():
        basic_name,global_name = basic_global_names(k,suffix_to_add=suffix_to_add)
        if suffix_to_filter_away is not None:
            curr_idx = basic_name.find(suffix_to_filter_away)
            basic_name= basic_name[:curr_idx-1]
        print(f"if {basic_name} is None:\n    {basic_name} = {global_name}")
        
def generate_default_assignment_block(param_dict,include):
    for k,v in param_dict.items():
        basic_name,global_name = basic_global_names(k)
        print(f"{global_name} = {v}")
def print_dict_with_suffix(my_dict,suffix):
    for k,v in my_dict.items():
        print(f"{k}_{suffix} = {v},")
    
def set_global_parameters_by_dict(module,
                                  param_dict,
                           global_suffix = "global",
                          verbose = False):
    for k,v in param_dict.items():
        local_name,global_name = modu.basic_global_names(k,global_suffix)
        try:
            setattr(module,global_name,v)
        except:
            print(f"Unable to set {k}")
            
        if verbose:
            print(f"{global_name} = {getattr(module,global_name)}")
            
def set_attributes_by_dict(module,
                           param_dict,
                           verbose = False):
    global_suffix = ""
    for k,v in param_dict.items():
        local_name,global_name = modu.basic_global_names(k,global_suffix)
        try:
            setattr(module,global_name,v)
        except:
            print(f"Unable to set {k}")
            
        if verbose:
            print(f"{global_name} = {getattr(module,global_name)}")
            
            
from . import numpy_utils as nu
from . import general_utils as gu
def collect_global_parameters_and_attributes_by_data_type(
    module,
    data_type,
    include_default = True,
    algorithms=None,
    output_types = None,
    algorithms_only= None,
    verbose = False):
    """
    PUrpose: To compile the dictionary to either
    set or output
    
    """
    if algorithms is not None:
        algorithms= nu.convert_to_array_like(algorithms)
    else:
        algorithms = []
        
    if output_types is None:
        output_types = output_types_global
    else: 
        output_types= nu.convert_to_array_like(output_types)
        
        
    
    p_list = dict()
    parameters_list = []
    
    if include_default and data_type != "default":
        total_data_types = ["default",data_type]
    else:
        total_data_types = [data_type]
    
    
    for dict_type in output_types:
        p_list[dict_type] = []
        for data_type in total_data_types:

            dict_name = f"{dict_type}_dict_{data_type}"
            
            if not algorithms_only:
                try:
                    curr_dict = getattr(module,dict_name).copy()
                except:
                    if verbose:
                        print(f"Unknown dict_name = {dict_name}")
                else:
                    if verbose:
                        print(f"Collecting {dict_name}")
                        print(f"curr_dict = {curr_dict}")
                    p_list[dict_type].append(curr_dict)

            for alg in algorithms:
#                 if data_type == "default":
#                     break
                dict_name = f"{dict_type}_dict_{data_type}_{alg}"
                try:
                    curr_dict = getattr(module,dict_name).copy()
                except:
                    if verbose:
                        print(f"Unknown dict_name = {dict_name}")
                else:
                    if verbose:
                        print(f"Collecting {dict_name}")
                        print(f"curr_dict = {curr_dict}")
                    p_list[dict_type].append(curr_dict)
    
    #compiling all the dicts
    if "global_parameters" in p_list:
        global_parameters_dict = gu.merge_dicts(p_list["global_parameters"])
    else:
        global_parameters_dict = {}
        
    if "attributes" in p_list:
        attributes_dict = gu.merge_dicts(p_list["attributes"])
    else:
        attributes_dict = {}
    
    return global_parameters_dict,attributes_dict
    
def extract_module_algorithm(module,
                             return_parameters = False,
                            verbose = False):
    """
    Will extract which model
    """
    if type(module) == dict:
        if verbose:
            print(f"\nalgorithms sent in dict")
        algorithms_local = module.get("algorithms",None)
        module = module["module"]
        algorithms_only = True

    elif nu.is_array_like(module,include_tuple=True):
        if verbose:
            print(f"\nalgorithms sent in array")
        algorithms_local = module[1]
        module = module[0]
        algorithms_only = True
    else:
        if verbose:
            print(f"\nNo algorithms per module set")
        algorithms_local = None
        algorithms_only = None
        
    if return_parameters:
        return module,algorithms_local,algorithms_only
    else:
        return module,algorithms_local
    

def set_global_parameters_and_attributes_by_data_type(
    module,
    data_type=None,
    algorithms=None,
    set_default_first = True,
    verbose = False
    ):
    
#     if set_default_first and data_type != "default":
#         if verbose:
#             print(f"Setting default first")
#         modu.set_global_parameters_and_attributes_by_data_type(module,
#                                                                data_type="default",
#                                                               set_default_first = False,
#                                                               verbose = False)
    if data_type is None:
        data_type = "default"
    
    module_list = nu.convert_to_array_like(module)
    
    
    for module in module_list:
            
        module, algorithms_local, algorithms_only = modu.extract_module_algorithm(
        module,
        return_parameters = True,
        )
        
        if algorithms_local is None:
            algorithms_local = algorithms
            
        if verbose:
            print(f"Setting data type dicts for module {module.__name__} "
                  f"with data_type = {data_type}, algorithms = {algorithms_local}")
        (global_parameters_dict,
        attributes_dict) = modu.collect_global_parameters_and_attributes_by_data_type(
        module=module,
        data_type=data_type,
        algorithms=algorithms_local,
        include_default = set_default_first,
        algorithms_only = algorithms_only,
        verbose = verbose
        )
        module.data_type = data_type
        module.algorithms = algorithms_local
        modu.set_global_parameters_by_dict(module,global_parameters_dict)
        modu.set_attributes_by_dict(module,attributes_dict)
        
    if verbose:
        print(f"\n--After Setting global parameters and attributes --")
        for m in module_list:
            m, algorithms_local = modu.extract_module_algorithm(
                m,
                return_parameters = False,
            )
        
            print(f"   module {m.__name__}: data_type = {m.data_type}, algorithms = {m.algorithms}")
        

"""
-----Example of how to use global parameters from synapse_utils-----

    attributes_dict_microns = dict(
        voxel_to_nm_scaling = alu.voxel_to_nm_scaling,
        data_fetcher = du
    )    

    global_parameters_dict_microns = dict()

    attributes_dict_h01 = dict(
        voxel_to_nm_scaling = hu.voxel_to_nm_scaling,
        data_fetcher = hdju
    )

    global_parameters_dict_h01 = dict()


    def set_global_parameters_and_attributes_by_data_type(data_type):
        import module_utils as modu 
        modu.set_global_parameters_and_attributes_by_data_type(syu,data_type)

    set_global_parameters_and_attributes_by_data_type("microns")

"""



def output_global_parameters_and_attributes_from_current_data_type(
    module,
    algorithms = None,
    verbose = True,
    lowercase = True,
    output_types = ("global_parameters"),
    include_default = True,
    algorithms_only = False,
    abbreviate_keywords = False,
    **kwargs
    ):
    
    if output_types is None:
        output_types = output_types_global
    
    module_list = nu.convert_to_array_like(module)
    total_dict_list = []
    for module in module_list:
        
        module, algorithms_local, algorithms_only_local = modu.extract_module_algorithm(
        module,
        return_parameters = True,
        )
        
        if algorithms_local is None:
            algorithms_local = algorithms
            
        if algorithms_only_local is None:
            algorithms_only_local = algorithms_only

        data_type = module.data_type
        
        if algorithms_local is None:
            algorithms_local = module.algorithms
            
        if verbose:
            print(f"module: {module.__name__} data_type set to {data_type}, algorithms = {algorithms_local}")
            

        (global_parameters_dict,
        attributes_dict) = modu.collect_global_parameters_and_attributes_by_data_type(
        module=module,
        data_type=data_type,
        algorithms=algorithms_local,
        include_default = include_default,
        output_types=output_types,
        algorithms_only = algorithms_only_local,
        verbose = verbose
        )

        total_dict = gu.merge_dicts([global_parameters_dict,attributes_dict])
        
        if lowercase:
            if isinstance(total_dict,dsu.DictType):
                total_dict = total_dict.lowercase()
            else:
                total_dict = {k.lower():v for k,v in total_dict.items()}
            
        total_dict_list.append(total_dict)
        
    final_dict = gu.merge_dicts(total_dict_list)    
    
    return final_dict


class ModuleDataTypeSetter:
    def __init__(
        self, 
        module = None,
        algorithms = None,
        ):
        self.module = module
        self.algorithms = algorithms

    def set_global_parameters_and_attributes_by_data_type(
        self,
        module=None,
        algorithms =  None,
        **kwargs
        ):
        
        if module is None:
            module= self.module
        if algorithms is None:
            algorithms = self.algorithms
            
        return modu.set_global_parameters_and_attributes_by_data_type(
            module=module,
            algorithms = algorithms,
            **kwargs
        )
    
    def output_global_parameters_and_attributes_from_current_data_type(
        self,
        module=None,
        algorithms =  None,
        **kwargs
        ):
        
        if module is None:
            module= self.module
        if algorithms is None:
            algorithms = self.algorithms
            
        return modu.output_global_parameters_and_attributes_from_current_data_type(
            module=module,
            algorithms = algorithms,
            **kwargs
        )
    
    
        
        



    
    
    
"""
Example of the fresh module functions:


# ------------- parameters for stats ---------------
import module_utils as modu
import general_utils as gu

global_parameters_dict_default = dict()
attributes_dict_default = dict()    


# ------- microns -----------
global_parameters_dict_microns = {}
attributes_dict_microns = {}


# --------- h01 -------------
global_parameters_dict_h01 = dict()
attributes_dict_h01 = dict()



data_type = "default"
algorithms = None
modules_to_set = [gf]

def set_global_parameters_and_attributes_by_data_type(dt,
                                                     algorithms_list = None,
                                                      modules = None,
                                                     set_default_first = True,
                                                      verbose=False):
    if modules is None:
        modules = modules_to_set
    
    modu.set_global_parameters_and_attributes_by_data_type(modules,dt,
                                                          algorithms=algorithms_list,
                                                          set_default_first = set_default_first,
                                                          verbose = verbose)
    
set_global_parameters_and_attributes_by_data_type(data_type,
                                                   algorithms)

def output_global_parameters_and_attributes_from_current_data_type(
    modules = None,
    algorithms = None,
    verbose = True,
    lowercase = True,
    output_types = ("global_parameters"),
    include_default = True,
    algorithms_only = False,
    **kwargs):
    
    if modules is None:
        modules = modules_to_set
    
    return modu.output_global_parameters_and_attributes_from_current_data_type(
        modules,
        algorithms = algorithms,
        verbose = verbose,
        lowercase = lowercase,
        output_types = output_types,
        include_default = include_default,
        algorithms_only = algorithms_only,
        **kwargs,
        )



Example on how to output: 
kwargs_dict = gu.merge_dicts([
    ctu.output_global_parameters_and_attributes_from_current_data_type(),
    au.output_global_parameters_and_attributes_from_current_data_type(),
    spu.output_global_parameters_and_attributes_from_current_data_type(
        algorithms = ["head_neck_shaft"],
        include_default=True,
        algorithms_only = True),
    syu.output_global_parameters_and_attributes_from_current_data_type()

])



Example of how to do abridge

"""

from pathlib import Path
import numpy as np

def all_modules_set_global_parameters_and_attributes(
    data_type,
    directory=Path("/meshAfterParty/meshAfterParty"),
    verbose = False,
    verbose_loop = False,
    ):
    """
    Purpose: To set the global parameters
    of all modules in a certain directory
    
    Ex: 
    import module_utils as modu
    modu.all_modules_set_global_parameters_and_attributes(
        "microns",
        verbose = True,
    )

    """
    p = Path("/meshAfterParty/meshAfterParty")
    mods = [k.stem for k in list(p.iterdir()) if k.suffix == ".py"]
    mods_set = []
    for k in mods:
        try:
            if verbose_loop:
                print(f"--Working on module {k}--")
            exec(f"import {k}")
            if verbose_loop:
                print(f"Accomplished import")
        except:
            if verbose_loop:
                print(f"Failed import ")
            continue

        success_set = False
        try:
            exec(f"{k}.set_global_parameters_and_attributes_by_data_type(data_type='{data_type}')")
        except:
            try:
                exec(f"{k}.set_global_parameters_and_attributes_by_data_type(dt='{data_type}')")
            except:
                if verbose_loop:
                    print(f"Failed setting plobal params ")
                    continue
            else:
                success_set = True
        else:
            success_set = True
        
        if (verbose_loop or verbose) and success_set:
            print(f"Accomplished setting global parameters")
            mods_set.append(k)
        

    if verbose:
        print(f"All modules set: {np.array(mods_set)}")

            
# ----------- more general module functions

def multiline_str_text(obj):
    """
    Purpose: To extract the inside string from a multiline str regex.Match object

    """
    return obj.groups()[1]

def multiline_str(
    filepath,
    beginning_of_line = True,
    verbose = False,
    return_text = False,
    above_first_func_def = False,
    ):
    """
    Purpose: find multiline strings in a module
    
    Ex: 
    multiline_str(
        filepath = "/python_tools/python_tools/example_re.py",
        verbose = True
    )
    """
    
    

    pattern = ru.multiline_str_pattern
    if beginning_of_line:
        pattern = ru.start_of_line_pattern + pattern

    pattern = re.compile(
        pattern,
        flags=re.DOTALL
    )

    data = filu.read_file(filepath)

    total_results = list(pattern.finditer(data))
    
    if above_first_func_def:
        idx = index_of_first_func_def(
            filepath = filepath
        )
        
        total_results = [k for k in total_results
                        if k.end() < idx]
    
    if return_text:
        total_results = [multiline_str_text(k) for k in total_results]
    if verbose:
        print(f"# of multi-line strings = {len(total_results)}")
    return total_results




from . import regex_utils as ru
from . import file_utils as filu
import regex as re

def import_pattern_str(
    start = None,
    beginning_of_line = True,
    modules = None,
    verbose = False,
    add_possible_comma = True
    ):
    
    if add_possible_comma:
        word_comb = ru.word_pattern_comma_space
    else:
        word_comb = ru.word_pattern
    
    
    if modules is None:
        modules = word_comb
    else:
        modules = f"({'|'.join(modules)})"
    
    if start is None:
        if beginning_of_line:
            start = ru.start_of_line_pattern
        else:
            start = ""
    
    import_str = (f"{start}("
        f"(?:import {modules} as {word_comb})"
        f"|(?:from {word_comb} import {modules} as {word_comb})"
        f"|(?:import {modules})"
        f"|(?:from {word_comb} import {modules})"  
        f"|(?:from [.]+{modules} import {word_comb})"
    ")"
    )
    
    if verbose:
        print(f"import_str = {import_str}")
    return import_str

def find_import_modules_in_file(
    filename=None,
    data = None,
    unique = True,
    verbose = False,
    verbose_import_pattern = False,
    pattern = None,
    modules = None,
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
    if data is None:
        data = filu.read_file(filename)
        
    #2) create the pattern to recognize imports
    if pattern is None:
        pattern = modu.import_pattern_str(
            modules=modules,
            beginning_of_line=beginning_of_line,
            verbose = verbose_import_pattern,
        )

    pattern = re.compile(pattern)
    finds = list(re.finditer(pattern,string=data))

    str_finds = [f.string[f.start():f.end()].replace('\n',"") for f in finds]

    if unique:
        str_finds = list(set(str_finds))

    if verbose:
        print(f"# of matches (unique = {unique}) = {len(str_finds)}")

    return str_finds

#from pathlib import Path
#import numpy as np
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

from . import pathlib_utils as plu
def modules_from_directory(
    directory,
    verbose = False,
    ignore_files = ("__init__",),
    ):
    """
    Purpose: Get all modules from directory

    Pseudocode:
    1) Get all of the py files in folder
    2) ignore the 
    """
    modules = plu.files_of_ext_type(
                        directory = directory,
                        ext = "py",
                        verbose = False
    )

    modules = [k.stem for k in modules if k.stem not in ignore_files]

    if verbose:
        print(f"# of modules = {len(modules)}")

    return modules


import regex as re
def index_of_first_func_def(
    string=None,
    filepath = None):
    """
    Purpose: Want to identify location of the first
    function definition

    Application: 
    1) only keep docstrings above the first function
    -> would eliminate the possibility of copying a function that is commented out
    """
    if string is None:
        string = filu.read_file(filepath)
    
    pattern=re.compile(fr"{ru.start_of_line_pattern}def ")
    curr_match = re.search(
        pattern = pattern,
        string=string,
    )

    if curr_match is not None:
        curr_match = curr_match.start()
    
    return curr_match


from pathlib import Path
from python_tools import system_utils as su
from python_tools import regex_utils as ru
from python_tools import module_utils as modu
from python_tools import file_utils as filu
from python_tools import string_utils as stru
from python_tools import package_utils as pku
from python_tools import pathlib_utils as plu
import numpy as np
import regex as re

def clean_module_syntax(
    filepath,
    output_path = None,
    overwrite = False,
    non_overwrite_suffix = "_replaced",
    verbose = False,
    packages = None,
    docstring_only_above_first_func_def = True,
    separator = f"",
    prefix_separator = "\n",
    skip_overwrite_suffix_files = True,
    ):
    """
    Part 0: Copy and paste document into new file if not overwrite

    Part 1: 

    Purpose: How to collect and rename the packages


    easy: these would just neeed to collect everything after the import statement
    f"(?:import {modules} as {word_comb})"
    f"|(?:from {word_comb} import {modules} as {word_comb})"
    f"|(?:import {modules})"
    f"|(?:from {word_comb} import {modules})"  

    harder: need to collect everything after the dot (add prefix with no space beforehand
    f"|(?:from [.]*{modules} import {word_comb})"

    Process for replacement:
    for each package
    1) find and replace the easy with (everything after import grouped)
    with "from package \1"
    2) find and replace the harder with (everything grouped after the dot)
    with "from package.\1"
    -- now for the relative replacement
    1) find and replace the easy (at the beginning of a line) by replacing
    from package with from [relative dots]
    2) find and replace hard (at beginning of the line) by replacing
    from package. with [relative dots]

    Part 2: Organization

    2) Get all the docstrings and replace in the file with nothing
    3) Find all of the module at the beginning of the line and replace with nothing
    (collect for organizing later)


    modules_dir = modu.find_import_modules_in_file(
        filepath,
        modules = modules,
        beginning_of_line = False
        )
    modules_dir

    4) Sort the modules into relative and not relative and itself
    5) Write the doc string and non-referencing modules 
    6) write last the referencing modules and then itself

    doc strings
    non-referencing modules
    #local veraibles
    rest of code
    referencing modules
    itself

    """
    if skip_overwrite_suffix_files and non_overwrite_suffix in str(filepath):
        return 
    
    
    if packages is None:
        packages = pku.user_packages
    
    
    filepath = Path(filepath)
    # 0) Create the output path (or set it to current path if overwrite)
    if not overwrite:
        if output_path is None:
            output_path = filepath.parents[0] / Path(f"{filepath.stem}{non_overwrite_suffix}{filepath.suffix}")
        su.copy_file(filepath,output_path)
    else:
        output_path = filepath

    output_path = Path(output_path)
        
    if verbose:
        print(f"output_path = {output_path}")
        
        
    # 1) Get all the information for the current module and all possible modules from all packages
    curr_mod = Path(filepath).stem
    curr_package_name = pku.package_from_filepath_and_package_list(
        filepath=filepath,
        packages=packages,
        return_package_name = True,
        verbose = verbose,
    )
    if verbose:
        print(f"curr_mod = {curr_mod}, curr_package_name = {curr_package_name}")
        
        
    pkg_to_module = {
        pku.package_name_from_path(k):pku.module_names_from_directories(k)
        for k in packages
    }
    
    
    for package_name,modules in pkg_to_module.items():
        if verbose:
            print(f"\n\n--- working on package: {package_name} ---")
        # 2) Run all of the module name replacements

        word_comb = ru.word_pattern
        modules_or = f"(?:{'|'.join(modules)})"

        # easy_import_pattern = (
        # f"(?:(import {modules_or} as {word_comb}))"
        # f"|(?:from {word_comb} (import {modules_or} as {word_comb}))"
        # f"|(?:(import {modules_or}))"
        # f"|(?:from {word_comb} (import {modules_or}))"  
        # )

        # this pattern will not match the proceeding and then in 
        # group 1 will match the 
        easy_import_pattern = (f"(?:from {word_comb} )?"
        f"(import {modules_or} as {word_comb}|import {modules_or})")

        easy_replacement = fr"from {package_name} \1"

        if verbose:
            print(f"-- easy initial replacement --")
        filu.file_regex_replace(
            pattern = easy_import_pattern,
            replacement = easy_replacement,
            filepath = output_path,
            overwrite_file = True,
            verbose = verbose
        )

        if verbose:
            print(f"-- harder initial replacement --")

        harder_import_pattern = f"(?:from [.]*({modules_or} import {ru.word_pattern_comma_space}))"
        harder_replacement = fr"from {package_name}.\1"
        filu.file_regex_replace(
            pattern = harder_import_pattern,
            replacement = harder_replacement,
            filepath = output_path,
            overwrite_file = True,
            verbose = verbose
        )

        if package_name == curr_package_name:
            if verbose:
                print(f"-- easier relative replacement --")

            easy_import_relative = f"({ru.start_of_line_pattern})from {package_name} "
            easy_replacement_relative = fr"\1from . "

            filu.file_regex_replace(
                pattern = easy_import_relative,
                replacement = easy_replacement_relative,
                filepath = output_path,
                overwrite_file = True,
                verbose = verbose
            )

            if verbose:
                print(f"-- harder relative replacement --")

            harder_import_relative = f"({ru.start_of_line_pattern})from {package_name}."
            harder_replacement_relative = fr"\1from ."

            filu.file_regex_replace(
                pattern = harder_import_relative,
                replacement = harder_replacement_relative,
                filepath = output_path,
                overwrite_file = True,
                verbose = verbose
            )
            
    #2) getting all the docstrings
    data = filu.read_file(output_path)
    multi_line_comm = modu.multiline_str(
        filepath = output_path,
        verbose = verbose,
        return_text=False,
        above_first_func_def = docstring_only_above_first_func_def,
    )

    range_list = [k.span() for k in multi_line_comm]

    data_doc = stru.remove_range_list(
        data,
        range_list=range_list,
        verbose = False,
    )

    all_doc = [modu.multiline_str_text(obj) for obj in multi_line_comm]
    
    
    
    #3) Getting all of the packages 
    finds = modu.find_import_modules_in_file(
        data = data_doc,
        unique = True,
        verbose = verbose,
        beginning_of_line = True,

    )

    finds = list(np.sort(finds))

    module_pattern = fr"{ru.start_of_line_pattern}({'|'.join(finds)})(?![0-9a-zA-z _\.\-:]+)"
    #print(f"module_pattern = {module_pattern}")
    data_doc_no_mod, count = re.subn(
        pattern = module_pattern,
        repl="",
        string=data_doc,
    )

    if verbose:
        print(f"# of modules replaced = {count}")

    #data_doc_no_mod = data_doc

    pkg_list = list(pkg_to_module.keys())

    non_pkg_mods = []
    pkg_mods = dict()
    own_mod = []

    for k in finds:
        set_flag = False
        if curr_mod in k:
            own_mod.append(k)
            continue
        if "from ." in k:
            if curr_package_name not in pkg_mods:
                pkg_mods[curr_package_name] = []
            pkg_mods[curr_package_name].append(k)
            continue

        for pkg in pkg_list:
            if pkg in k:
                if pkg not in pkg_mods:
                    pkg_mods[pkg] = []
                pkg_mods[pkg].append(k)
                set_flag = True
                break
        if not set_flag:
            non_pkg_mods.append(k)


    non_pkg_mods_str = "\n".join(non_pkg_mods)
    own_mod_str = "\n".join(own_mod)
    pkg_mods_str = "\n\n".join([f"#--- from {pkg} ---\n" + "\n".join(m)
                               for pkg,m in pkg_mods.items() ])
    
    #get rid of any old package documentation (in case run documentation again)
    for pkg in pkg_mods.keys():
        data_doc_no_mod = data_doc_no_mod.replace(f"#--- from {pkg} ---\n","")

    if verbose:
        print(non_pkg_mods_str)
        print(f"\n")
        print(pkg_mods_str)
        print(f"\n")
        print(own_mod_str)
        
        
    # 4) Creating the final module string


    if len(all_doc) > 0:
        all_doc_str = "\n\n".join(all_doc)
        all_doc_str = f"'''\n{all_doc_str}\n'''{separator}"
    else:
        all_doc_str = ""

    if len(non_pkg_mods_str) > 0:
        non_pkg_mods_str = f"{prefix_separator}{non_pkg_mods_str}{separator}"
    if len(pkg_mods_str) > 0:
        pkg_mods_str = f"{prefix_separator}{pkg_mods_str}{separator}"
    if len(own_mod_str) > 0:
        own_mod_str= f"{prefix_separator}{own_mod_str}"

    final_data = (
        all_doc_str +
        non_pkg_mods_str + 
        data_doc_no_mod +
        pkg_mods_str + "\n" +
        own_mod_str
    )
    
    
    # 5) Outputting final string
    filu.write_file(
        output_path,
        final_data,
        replace=True,
    )

    
import regex as re
def package_from_import_string(
    string,
    ):
    """
    Purpose: find the package 
    from the import statement using
    regex
    """
    pattern = fr"\s*(?:from|import) ([a-zA-Z_]+|.)"

    package = re.search(
        pattern=pattern,
        string=string,
    ).groups()[0]

    return package
    
    
from python_tools import module_utils as modu
from python_tools import package_utils as pku
from pathlib import Path
import numpy as np
from python_tools import numpy_utils as nu

def package_imports_from_files(
    files = None,
    directory = None,
    verbose = False,
    ):
    """
    Purpose: Find all of the modules imported
    from all the files in a directory and 
    compile a unique list

    Pseudocode: 
    1) Get all of the modules in the directory
    2) For each module, get a list of the imports
    3) restrict the name of the imports to only beginning important part
    4) make a unique list from all of the lists
    
    Ex: 
    modu.package_imports_from_files(
        directory = "/graph_tools/graph_tools/",
    )
    """
    if files is None:
        files = pku.module_files_from_directory(directory)

    files = nu.to_list(files)

    all_imports = []
    for filepath in files:
        if verbose:
            print(f"-- working on {filepath} --")
        import_statements = modu.find_import_modules_in_file(
            filename = filepath,
            beginning_of_line=False,
        )
        if len(import_statements) > 0:
            all_imports += import_statements

    #3) restrict the name of the imports to only beginning important part
    all_imports = [
        modu.package_from_import_string(k) for k in all_imports
    ]

    all_imports = list(np.sort(list(set(all_imports))))
    all_imports = [k for k in all_imports if k != "."]
    return all_imports
    

    
    
    
#from python_tools import file_utils as filu


            

    


