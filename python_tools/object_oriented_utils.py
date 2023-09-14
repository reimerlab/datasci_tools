"""
To provide information and functionality for object oriented programming
"""

def example_slots():
    """
    Purpose: if don't want to be able to add just any named attribute
    to an object dynamically (becuase normally can), can specify all possible
    attributes you might want
    
    Why? So that the instance.__dict__ doesn't grow large
    """
    
    class S(object):

        __slots__ = ['val']

        def __init__(self, v):
            self.val = v


    x = S(42)
    print(x.val)
    
    # this would result in an error
    x.new = "not possible"
    
