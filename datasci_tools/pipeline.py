import copy

class StageProducts:
    def __init__(
        self,
        *args,
        **kwargs):
        if len(args) > 0:
            if isinstance(args[0],self.__class__):
                kwargs.update(args[0].export()) 
            elif type(args[0]) == dict:
                kwargs.update(args[0])
            elif type(args[0]) == None:
                pass
        for k,v in kwargs.items():
            setattr(self,k,v)
            
    def export(self):
        export_dict = {k:getattr(self,k) for k in dir(self)
                       if k[:2] != "__" and "bound method" not in str(getattr(self,k))
                       #and not funcu.isfunction(getattr(self,k))
                       }
        return export_dict
    
    def update(self,*args,**kwargs):
        if len(args) > 0:
            if isinstance(args[0],self.__class__):
                kwargs.update(args[0].export())
            elif type(args[0]):
                kwargs.update(args[0])
            else:
                pass
            
        for k,v in kwargs.items():
            setattr(self,k,v)
            
    def __getitem__(self,k):
        return getattr(self,k)
    
    def __setitem__(self,k,v):
        return setattr(products,k,v)
    
    def __str__(self):
        ret_str=""
        for k,v in self.export().items():
            ret_str+=(f"    {k}:{v}\n")
        return ret_str
    
    def __contains__(self,key):
        return key in self.export()
    
    def __iter__(self,*args,**kwargs):
        return self.export().__iter__(*args,**kwargs)
    
    def get(self,k,*args):
        return self.export().get(k,*args)
    
    def values(self):
        return self.export().values()
    def keys(self):
        return self.export().keys()
  
            
default_stage = "NoStage"
class PipelineProducts:
    def __init__(
        self,
        *args,
        **kwargs
        ):
        
        if len(args) > 0:
            if args[0] is None:
                pass
            elif type(args[0]) == type(self):
                kwargs.update(args[0].export())
            elif type(args[0]) == dict:
                kwargs.update(args[0])
            
        self.products = {k:StageProducts(v) for k,v in kwargs.items()}
        # for k,v in self.products.items():
        #     if type(v) != StageProducts:
        #         raise Exception(f"{k} is not a StageProducts object")
        
    def __str__(self,):
        gu.print_nested_dict(self.export())
        return ""
        
        # ret_str = ""
        # for k,v in self.products.items():
        #     ret_str += f"{k}\n{str(v)}\n"
        # return ret_str
    
    def __getitem__(self,k):
        return getattr(self,k)
    
    def __setitem__(self,k,v):
        self.products[k] = v
        
                
    def export(self):
        return {k:v.export() for k,v in self.products.items()}
    
    @property
    def stages(self):
        return list(self.products.keys())
    
    def values(self):
        return self.products.values()
    def keys(self):
        return self.products.keys()
    def items(self):
        return self.products.items()
    
    def set_stage_attrs(
        self,
        attr_dict,
        stage,
        clean_write = False,
        **kwargs
        ):
        
        attr_dict.update(kwargs)
        
        if clean_write or stage not in self.products:
            self.products[stage] = StageProducts(attr_dict)
        else:
            self.products[stage].update(attr_dict)
            
    def set_attr(
        self,
        attr,
        value=None,
        stage = None,
        **kwargs
        ):
        
        if type(attr) == dict:
            attr_dict = attr
        else:
            attr_dict = {attr:value}
        
        for k,v in attr_dict.items():
            if stage is None:
                stage = self.stage_from_attr(attr)
                
            if stage is None:
                stage = default_stage
                
            self.set_stage_attrs(
                {k:v},
                stage = stage,
                clean_write = False,
                **kwargs
            )
        
    def set_attrs(
        self,
        attr_dict,
        ):
        
        self.set_attr(
            attr_dict,
            **kwargs
        )

    def get_attr(
        self,
        attr,
        stages = None,
        verbose = False,
        default_value = None,
        error_on_not_found = False,
        return_stage = False,):
        """
        Purpose: Want to get a stage attribute but not 
        necessarily know what the stage name is called

        """
        if stages is None:
            stages = [k for k in self.stages[::-1]]
            
        
        for st in stages:
            v = self.products[st]
            if hasattr(v,attr):
                if verbose:
                    print(f"Found {attr} in stage {st}")
                ret_value = getattr(v,attr)
                if return_stage:
                    return ret_value,st
                else:
                    return ret_value
        
        not_found_str = f"{attr} was not found in any stage"
            
        if error_on_not_found:
            raise Exception(f"{not_found_str}")
        elif verbose:
            print(f"{not_found_str}")
        else:
            pass
            
        if return_stage:
            return default_value,None
        else:
            return default_name
        
    def stage_from_attr(self,attr):
        _, stage = self.get_attr(attr,return_stage=True)
        
        return stage
    
    def __contains__(self,k):
        if k in self.products:
            return True
        if self.stage_from_attr(k) is not None:
            return True
        else:
            return False
        
    def get_attrs(
        self,
        attrs,
        stages = None,
        verbose = False,
        default_value = None,
        error_on_not_found = False,
        return_stage = False,
        ):
        attrs = nu.to_list(attrs)
        
        attrs_values = dict()
        stage_values = dict()
        
        for k in attrs:
            att_val,st_name = self.get_attr(
                k,
                stages = stages,
                verbose = verbose,
                default_value = default_value,
                error_on_not_found = error_on_not_found,
                return_stage = True,)
            attrs_values[k] = att_val
            stage_values[k] = st_name
            
        if return_stage:
            return attrs_values,stages
        else:
            return attrs_values
    
    def __getattr__(self, name):
        #print(f"self = {self.__repr__}")
        #print(f"name = {name}")
        if name[:2] == "__":
            raise AttributeError(name)
        
        if name in list(self.products.keys()):
            return self.products[name]
        else:
            try:
                return self.get_attr(
                    name,error_on_not_found=True
                )
            except:
                return self.__getattribute__(name)
            
    def get(self,key,*args):
        """
        To mirror the get function in dictionaries
        but that goes a little recursive
        """
        if key in self.products:
            #print(f"Top level")
            return self.products.get(key)
        else:
            for sk,sv in self.products.items():
                if key in sv:
                    #print(f"Low level")
                    return sv[key]
                else:
                    continue
        if len(args) > 0:
            return args[0]
            #print(f"Default level")
        else:
            raise Exception(f"Could not find value {key}")
        
    def delete_stage(self,stage):
        try:
            del self.products[stage]
        except:
            pass
        
    def __iter__(self,*args,**kwargs):
        return self.products.__iter__(*args,**kwargs)
            
    

from . import function_utils as funcu
from . import numpy_utils as nu
from . import general_utils as gu
