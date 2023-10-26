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
    from datasci_tools import string_utils as stru
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
    from datasci_tools import string_utils as stru
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
    from datasci_tools import string_utils as stru
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

def replace_patterns(
    string,
    patterns,
    replacement = "",
    ):
    patterns = nu.to_list(patterns)
    for p in patterns:
        string = string.replace(p,replacement)
    return string

def eliminate_patterns(
    string,
    patterns,
    ):
    return replace_patterns(string,patterns)

#from datasci_tools import string_utils as stru

def strip_whitespace(
    string,
    whitespace_char = None,
    additional_whitespace_char = None,
    leading = True,
    trailing = True,
    verbose = False,):
    """
    Purpose: Want to eliminate leading and trailing whitespace 
    from a string (whitespace = spaces, newlines, tabs)

    Pseudocode:
    For beginning and end
        1) Create a regex pattern to capture all leading or trailing
        white space
        2) Eliminate that group from the original string
        
    Ex: 
    from datasci_tools import string_utils as stru
       
    stru.strip_whitespace(
        string = "     \t\n    hello there             \t \n\n\n",
        #string = "   hello ",
        verbose = True,
    )

    """
    if whitespace_char is None:
        whitespace_char = (
        " ",
        "\t",
        "\n",
        reu.start_of_file_pattern,
        reu.end_of_file_pattern
        )
        
    if additional_whitespace_char is not None:
        whitespace_char = list(whitespace_char) + nu.to_list(additional_whitespace_char)
    
    trials = []
    if leading:
        trials.append((0,1))
    if trailing:
        trials.append((-1,-1))
    
    for start_idx,incrementer in trials:
        for i in range(len(string)):
            curr_idx = start_idx+incrementer*i
            curr_char = string[curr_idx]
            if curr_char not in whitespace_char:
                if verbose:
                    print(f"Non-whitespace char ({curr_char}) at idx = {curr_idx}")
                if start_idx == 0:
                    string = string[curr_idx:]
                else:
                    if curr_idx < -1:
                        string = string[:curr_idx+1]
                break
    return string


#--- from datasci_tools ---
from . import numpy_utils as nu
from . import regex_utils as reu

from . import string_utils as stru