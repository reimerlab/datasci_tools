'''


Purpose: To implement an enum (non-native) in python where
the list has an order to them

Purpose of enum: named constants


'''
from enum import Enum
def example():
    # class syntax
    class Color(Enum):
        RED = 1
        GREEN = 2
        BLUE = 3

        
    #examle use cases of Color enum
    some_var = Color.RED
    some_var in Color # will print yes
    
    
    Color.BLUE.name
    # >> 'Blue'
    Color.RED.value
    # >> 1

