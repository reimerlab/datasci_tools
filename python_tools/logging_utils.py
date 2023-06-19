"""
Covers the basics of the logging module (built in python)

---------source 1--------------
link: https://towardsdatascience.com/logging-in-python-a1415d0b8141

Standard levels (WARNING is the default level)
┌──────────┬───────┐
│  Level   │ Value │
├──────────┼───────┤
│ CRITICAL │    50 │
│ ERROR    │    40 │
│ WARNING  │    30 │
│ INFO     │    20 │
│ DEBUG    │    10 │
└──────────┴───────┘

Purpose of module: without changing any code except the basicConfig function
1) To make print statements really easy to turn off and on with different levels
2) To output any print/debug statements to a file without changing, 
3) print out other things (like time and traceback calls)

When would you use it? 
1) debugging, 
2) usage monitoring
3) performance monitoring

How to use: 
1) put "logging.[level]('some message in code')"
2) set "logging.basicConfig(level=logging.LEVEL)"
- only that level and above will be printed or executed
3) can set more args in basicConfig to 
    - print to text 
    - change output formatting

# -- named logger vs root logger --

1) How to use the root logger
logging.debug("") OR logger = logging.getLogger(); logger.debug("")

2) Using a named logger so that you are able to tell where logging came from
logger = logging.getLogger('spam_application'); logger.debug("")


To automatically create a specific logger for every distinct module use:
logger = logging.getLogger(__name__)
"""

import logging


def examples():
    # -- Ex: Basic
    import logging

    logging.debug('Debug message')
    logging.info('Info message')
    logging.warning('Warning message')
    logging.error('Error message')
    logging.critical('Critical message')
    
    
    # -- Ex: output to file (the last 2 messages would print)
    import logging

    logging.basicConfig(filename='sample.log', level=logging.INFO)

    logging.debug('Debug message')
    logging.info('Info message')
    logging.error('Error message')
    
    
    # -- Ex: changing formatting
    '''
    default formatting = %(levelname)s:%(name):%(message)s
    '''
    
    FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'

    logging.basicConfig(format=FORMAT, level=logging.INFO)
    logging.info('Info message')
    
    
    # -- Ex: change date formatting
    FORMAT = '%(asctime)s:%(name)s:%(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, 
                        level=logging.INFO, 
                        datefmt='%Y-%b-%d %X%z')

    logging.info('Info message')
    
    
    # -- Ex: logging an exception with a traceback
    # - way 1: with error and a flag set
    try:
        5/0

    except:
        logging.error('Exception occured', exc_info=True)
        
        
    # - way 2: with exception class
    try:
        5/0

    except:
        logging.exception('Exception occured')