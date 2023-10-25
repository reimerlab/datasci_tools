'''



Purpose: functions for quick hashing




'''
import base64
import hashlib

#import base64
#import hashlib

def hash_str(string,max_length = 10):
    """
    Purpoose: To hash a string and truncate to a certain
    length
    
    Example:
    from datasci_tools import hash_utils as shu
    shu.hash_str("The quick brown fox")
    """
    hasher = hashlib.sha1((string).encode('utf-8'))
    x = base64.urlsafe_b64encode(hasher.digest()[:max_length])
    x = x.decode('UTF-8')
    if len(x) > max_length:
        return x[:max_length]
    else:
        return x
"""
test comment
test 2
test 5
"""
#from datasci_tools import hash_utils as hashu




from . import hash_utils as hashu