
from typing import List

'''
module for helping create type hinting

documentation: https://docs.python.org/3/library/typing.html

Ex: 

def get_install_requires(filepath=None) -> List[str]:
    if filepath is None:
        filepath = "./"
    """Returns requirements.txt parsed to a list"""
    fname = Path(filepath).parent / 'requirements.txt'
    targets = []
    if fname.exists():
        with open(fname, 'r') as f:
            targets = f.read().splitlines()
    return targets


'''

# example function with type annottions
def headline(text: str, centered: bool = False) -> str:
    if not centered:
        return f"{text.title()}\n{'-' * len(text)}"
    else:
        return f" {text.title()} ".center(50, "o")
    
    
from typing import Union
def foo(client_id: str) -> Union[list, bool]:
    pass


