'''


Utility functions for manipulating strings



'''
import difflib

def abbreviate_phrase(
    s,
    separating_character = "_",
    max_word_len = 2,
    max_phrase_len = 20,
    include_separating_character = True,
    verbose = False,
    ):
    """
    Purpose: To abbreviate a string
    with a cetain number of characters
    for each word seperated by a 
    certain character
    
    Ex: 
    from python_tools import string_utils as stru
    stru.abbreviate_phrase(
        s = "ais_syn_density_max_backup_excitatory",
        verbose = True,
    )

    """

    if verbose:
        print(f"Original Length: {len(s)}")

    split_chars = s.split(separating_character)

    if not include_separating_character:
        separating_character = ""

    comb_s = separating_character.join([k[:max_word_len]
                for k in split_chars])[:max_phrase_len]

    if verbose:
        print(f"New length: {len(comb_s)}")
        
    return comb_s


#import difflib

def str_overlap(s1, s2):
    """
    Ex: 
    from python_tools import string_utils as stru
    stru.str_overlap("my name is Brendan","helloBrend")
    """
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return s1[pos_a:pos_a+size]


def is_int(s):
    return s.isdigit()

def number_to_letter(number,upper = False):
    """
    Int to letter string: 
    
    Ex: 
    number_to_letter(3)
    """
    return_value =  chr(ord('@')+(number+1)).lower()
    if upper:
        return return_value.upper()
    else:
        return return_value
    
def letter_to_number(letter):
    """
    letter to string string: 
    
    Ex: 
    letter_to_number(3)
    """
    return ord(letter) - ord('a')

def example_curly_braces_inside_fstring():
    """
    https://stackoverflow.com/questions/42521230/how-to-escape-curly-brackets-in-f-strings
    """
    x = "hi"
    print(f"{x} is better than {{hello}}")
    
def remove_range_list(
    string,
    range_list,
    verbose = False):
    
    """
    Ex: 
    from python_tools import string_utils as stru
    stru.remove_range_list(
        string = 'geeksforgeeks is best for geeks',
        range_list = [(3, 6), (7, 10), (14, 17)],
        verbose = True
    )
    """
    
    return nu.remove_range_list(
        obj=string,
        range_list=range_list,
        remove = True,
        verbose = verbose
    )

def keep_range_list(
    string,
    range_list,
    verbose = False):
    
    return nu.remove_range_list(
        obj=string,
        range_list=range_list,
        remove = False,
        verbose = verbose
    )


    

#from python_tools import string_utils as stru


#--- from python_tools ---
from . import numpy_utils as nu

from . import string_utils as stru
