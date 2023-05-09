from collections import UserDict

def parse_type_from_entry(v,default_type=None):
    if type(v) == list or type(v) == tuple:
        if type(v[0]) == type:
            type_idx = 0
            value_idx = 1
        else:
            type_idx = 1
            value_idx = 0
        data_v = v[value_idx]
        type_v = v[type_idx]
    else:
        data_v = v
        type_v = default_type
        
    return data_v,type_v
        

class DictType():
    """
    Purpose: To have a dictionary like object
    that also stores the default typs of the data
    if need be
    
    Ex: 
    import data_structure_utils as dsu
    my_obj = dsu.DictType(hello=("x",str),hi=(int,5),yes=6)
    my_obj2 = dsu.DictType(hello=("xy",str),his=(int,5),yes=(10,bool))
    my_obj

    from python_tools import general_utils as gu
    output_obj = gu.merge_dicts([my_obj,my_obj2,{}])
    output_obj
    
    """
    def __init__(self,*args,
                 **kwargs):
        self.default_type = kwargs.pop("default_type",None)
        
        if len(args)>0:
            curr_arg = args[0]
            if isinstance(curr_arg,self.__class__):
                self.default_types = curr_arg.default_type
                self._dict = curr_arg._dict.copy()
                self._types = curr_arg._types.copy()
                return 
            elif type(curr_arg) == dict:
                kwargs = curr_arg
            
        data_dict,types_dict = self.parse_types_from_dict(**kwargs)
        self._dict=data_dict
        self._types = types_dict
    
    def __delitem__(self, key):
        del self._dict[key]
        del self._types[key]
    
    
            
    def parse_types_from_dict(self,
                              parse_verbose = False,
                             **kwargs):
        data_dict = dict()
        type_dict = dict()
        for k,v in kwargs.items():
            data_v,type_v = parse_type_from_entry(v,self.default_type)
            data_dict[k] = data_v
            if type_v is not None:
                type_dict[k] = type_v
        
        if parse_verbose:
            print(f"data_dict = {data_dict}, type_dict = {type_dict}")
        return data_dict,type_dict
    
    def __getitem__(self,key):
        return self._dict[key]
    
    def __setitem__(self,key,value):
        data_v,type_v = parse_type_from_entry(value)
        self._dict[key] = data_v
        if type_v is not None:
            self._types[key] = type_v
        
    def __delitem__(self, key):
        del self._dict[key]
        del self._types[key]
        
    def __iter__(self):
        return iter(self._dict)    
    
    def __len__(self):
        return len(self._dict)
    
    def update(self,B):
        B_obj = self.__class__(B)
        self._dict.update(B_obj._dict)
        self._types.update({k:v for k,v in B_obj._types.items() if v is not None})
        
    def copy(self):
        return self.__class__(self)
        return new_obj
    
    def __str__(self):
        total_str = "{"
        for k,v in self._dict.items():
            total_str += f"'{k}': {v} ({self._types.get(k,self.default_type)}), "
        total_str += "}"
        return total_str
    
    def __repr__(self):
        return str(self)
    
    def keys(self):
        return self._dict.keys()
    def values(self):
        return self._dict.values()
    def items(self):
        return self._dict.items()
    
    def lowercase(self):
        cp = self.copy()
        cp._dict = {k.lower():v for k,v in cp._dict.items()}
        return cp
    
    def asdict(self):
        return self._dict
        
    
    

from python_tools import data_struct_utils as dsu