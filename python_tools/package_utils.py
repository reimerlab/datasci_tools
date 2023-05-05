import pathlib_utils as plu
import numpy_utils as nu
import file_utils as filu
import io


def prefix_module_imports_in_files(
    filepaths,
    modules_directory = "../python_tools",
    modules = None,
    prefix = "from . ",
    overwrite_file = False,
    output_filepath = None,# "text_revised.txt",
    verbose = False,
    ignore_files = ["__init__"],
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

    #1) if given a directory: get a list of the module names
    if modules is None:

        modules_directory = nu.to_list(modules_directory)
        modules = []

        for directory in modules_directory:

            if verbose:
                print(f"--Getting files from {directory}")

            modules += plu.files_of_ext_type(
                directory = directory,
                ext = "py",
                verbose = verbose
            )
    else:
        modules = [Path(k) for k in nu.to_list(modules)]

    modules = [k.stem for k in modules if k.stem not in ignore_files]

    #2) Construct a regex pattern ORing the potential list
    pattern = f"(import ({'|'.join(modules)}))"
    #replacement = fr"{prefix} import \1"

    filepaths = nu.to_list(filepaths)

    #4) write to a new file or old file
    for f in filepaths:
        if verbose:
            print(f"--- Working on file: {f}")
        filu.file_regex_add_prefix(
            pattern=pattern,
            prefix=prefix,
            filepath=f,
            overwrite_file = overwrite_file,
            output_filepath = output_filepath,# "text_revised.txt",
            verbose = verbose,
            regex = True,
        )


import package_utils as pku