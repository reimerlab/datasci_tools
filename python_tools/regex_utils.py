import re
import re 

"""
Note: 
- * only works at the beginning of string

Rules:
. any single character
.* zero or more characters but scan all of them
.+ one or more characters but scan all of them
.+? zero or more characters, scan up to all of them but be wishy washy about it
(.*) zero or more characters, scan all of them, but collect the whole thing as a field and then do nothing with it.


#--- cheat sheet ----: https://www.rexegg.com/regex-quickstart.html
. : any singl character
[] one of characters in brackets
[]+: 1 or more of word wherever evercharacter is one category in the brackets

\b: word boundary
\w: word
\W: not word
\d: digit
\D: not digit
\s: whitespace
^: beginning of string
$: end of string
|: means or


\d{3}: 3 digits
\d?: optionally 0 or 1 digit
\d{0,3}: 0 to 3 digits

[A-Za-z0-9_\s]+ : will match a length 1 or longer string with only characters_numbers and underscores and white space

Rule 1: Surround expression with () to designate it as a group

# -------- how to use wildcard character in teh middle
dict_type = "global_parameters"
algorithm = "split"
data_type = "h01"
search_string = fr"{dict_type}.*{algorithm}.*{data_type}"
test_str = "global_parameters_hi_split_h01"
#from python_tools import regex_utils as ru
ru.match_substring_in_str(search_string,
            test_str)


"""

"""
Exaples of searches-----------------
Basic process: 
1) Compile a patter
2) run find iter or search
3) Do something with the results

------------Example 1: shows how to find start and end ------------
#import re
s = "hello there 12345 hig pig"
pattern = re.compile(r"\b(\w{3})\b")
s_find = pattern.finditer(s)

for st in s_find:
    print(st.start(),st.end())
    print(st.span())
    print(st.groups())
    
    
--------- Example 2: How to replace expression with function ------
#import re
s = "hello there 12345 hig pig"
pattern = re.compile(r"\b(\w{3})\b")

def convert_func(match_obj):
    return f"--this was {match_obj.group(1)} replaced--"

res_str = pattern.sub(convert_func,s)


------ Example 3: Showing complex replacement --------
'''
Accomplishes the following change
before replace:
 (x == MAX( x )) or (x = 6)
after replace:
 (x == (SELECT MAX( x ) FROM df)) or (x = 6)

'''

verbose = True
s = "(x == MAX( x )) or (x = 6)"

if verbose:
    print(f"before replace:\n {s}")

def convert_func(match_obj):
    return f"(SELECT {match_obj.group(1).upper()}({match_obj.group(2)}) FROM df)"


sql_pattern = re.compile(fr"({func_name}|{func_name.lower()}|{func_name.upper()})\(([A-Za-z0-9_\s]+)\)")
s = sql_pattern.sub(convert_func,s)

if verbose:
    print(f"after replace:\n {s}")
    
    
# ------- searching for something but only replacing a certain part ----

Ex: Will 
s = dotmotif_str
s = "6P-PT  -> hellow"
pattern = (
    "([a-zA-Z0-9])-([a-zA-Z0-9])")
re.sub(pattern,r"\1\2",s)

# --- captured groups vs uncaptured groups
(?: something) this is an uncaptured group, where group something together but not save for named
():captured group

"""
start_of_file_pattern = r"\A"
end_of_file_pattern = r"\Z"

start_of_line_pattern = fr"(?:{start_of_file_pattern}|\n)"
multiline_str_pattern = r"""(['"])\1\1(.*?)\1{3}"""

word_pattern = "[a-zA-Z._]+"


def multiple_replace(
    text,
    dict_map=None,
    pattern = None):
    """
    Purpose: To replace multiple strings with a dictionary apping
    
    Ex: 
    from python_tools import regex_utils as ru
    query = "u in [1,2,3,4]"
    dict_mapping = dict(u="v",v="u")
    ru.multiple_replace(query,dict_mapping)
    """
    # Create a regular expression  from the dictionary keys
    if dict_map is not None:
        regex = re.compile("(%s)" % "|".join(map(re.escape, dict_map.keys())))

        # For each match, look-up corresponding value in dictionary
        text = regex.sub(lambda mo: dict_map[mo.string[mo.start():mo.end()]], text)
    
    if pattern is not None:
        pass
    
    return text
        

def all_match_substring_in_str(substring,expression):
    z = re.match(substring,expression)
    return z

def match_substring_in_str(substring,expression):
    z = re.match(substring,expression)
    if z:
        return True
    else:
        return False
    
def substr_from_match_obj(match_obj):
    return match_obj.string[match_obj.span()[0]:match_obj.span()[1]]

def sub_str_for_pattern(
    s,
    pattern,
    replacement):
    return re.sub(pattern,replacement,s)

def sub_str_for_pattern_with_count(
    s,
    pattern,
    replacement):
    return re.subn(pattern,replacement,s)

def match_pattern_in_str(
    string,
    pattern,
    return_one = False,
    verbose = False
    ):
    """
    Purpose: To find the string that
    matches the pattern compiled
    """

    pattern = re.compile(pattern)
    s_find = pattern.finditer(string)
    found_strings = []
    for st in s_find:
        found_strings.append(string[st.start():st.end()])
        
    if verbose:
        print(f"found_strings = {found_strings}")

    if return_one:
        return found_strings[0]
    else:
        return found_strings
    


from . import regex_utils as ru