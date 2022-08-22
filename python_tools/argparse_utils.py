"""
Purpose: Go through the specifics 
of how to enable commmand line interface for a python script

Tutorial: https://realpython.com/command-line-interfaces-python-argparse/
"""

import argparse

general_notes = """
- when using argparse get the help (-h) for free and will automatically tell you the variables you are missing
"""


argument_options_notes= """
An argument is a single part of a command line, delimited by blanks.
An option is a particular type of argument (or a part of an argument) that can modify the behavior of the command line.
A parameter is a particular type of argument that provides additional information to a single option or command.


Ex: 
$ ls -l -s -k /var/log

-ls: the name of the command you are executing
-l: an option to enable the long list format
-s: an option to print the allocated size of each file
-k: an option to have the size in kilobytes
/var/log: a parameter that provides additional information (the path to list) to the command
"""

argparse_arguments_notes ="""
    metavar: provides different names for optional arguments in help mesages
        ex: (see how it replaces the bar variable )
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--foo', metavar='YYY')
        >>> parser.add_argument('bar', metavar='XXX')
        >>> parser.parse_args('X --foo Y'.split())
            Namespace(bar='X', foo='Y')
        >>> parser.print_help()
            usage:  [-h] [--foo YYY] XXX
    
    optional arguments start with - or -- while positional dont
    
    default: sets the default value
    
    type: can set the type of the arguments
    
    required : wether argument is required or not
    
    help: gives a brief description of what it does
    
    dest: what you want the argument to be named when stored
"""

def example_basic_argparse_test():

    my_parser = argparse.ArgumentParser(description="List the contents of a folder",
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    

    my_parser.add_argument('Path',metavar='path',type=str,help='the path to list')

    args = my_parser.parse_args("curr_path".split())
    input_path = args.Path

    input_path
    
    
def print_help_str(parser):
    parser.print_help()