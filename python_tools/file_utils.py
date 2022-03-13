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