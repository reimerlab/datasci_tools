"""
Utility functions for manipulating strings

"""

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
    import string_utils as stru
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