from numpy import *
from numpy import array as _arr

def array(*args,**kwargs):
    try:
        x = _arr(*args,**kwargs)
    except:
        kwargs["dtype"] = "object"
        x = _arr(*args,**kwargs)
    return x