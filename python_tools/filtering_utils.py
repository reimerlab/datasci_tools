
import networkx as nx
from . import numpy_dep as np
"""
Purpose: Will aid in filtering a set
of objects based on a list of functions or statistics
computed over that object



Example on how to query objects:

#from python_tools import filtering_utils as flu
C = flu.Comparator(shaft_candidates_filtered,neuron_obj=neuron_obj)

def n_branches_from_candidate(candidate):
    return len(candidate["branches"])

C.compute_node_properties(node_functions = [n_branches_from_candidate])
C.objects_from_node_query("n_branches_from_candidate > 2")


Good example of this usage is: 
- apical_utils.filter_apical_candidates_to_one

"""

#from python_tools import networkx_utils as xu
#import networkx as nx
#from . import numpy_dep as np
#from python_tools import tqdm_utils as tqu
default_attribute_name = "obj"
node_arguments_name = "arguments"
#from python_tools import numpy_utils as nu
class Comparator:
    """
    Purpose: To store objects in a relational way and to
    1) Compute properties of the object
    2) Store relational properties between objects
    3) Query for subsets of the objects
    
    """
    def __init__(self,
                 objects,
                 attribute_name=default_attribute_name,
                 function_args=None,
                 object_attributes = None,
                 **kwargs):
        self._objects = objects
        self.function_args = function_args
        self.attribute_name = attribute_name
        self.G = flu.G_from_objects(objects,
                                    attribute_name=self.attribute_name,
                                   object_attributes=object_attributes)
    @property
    def objects(self):
        return flu.objects_from_G(self.G,attribute_name=self.attribute_name)
    
    @property
    def edge_df(self):
        return xu.edge_df(self.G,)
    
    @property
    def node_df(self):
        return xu.node_df(self.G,properties_to_exclude=[default_attribute_name,
                                                       node_arguments_name])
    
    def compute_node_properties(self,node_functions,
                               object_argument_name=None,
                               verbose=False):
        flu.compute_node_properties(self.G,node_functions,
                            object_argument_name=object_argument_name,
                            default_arguments=self.function_args,
                           verbose = verbose)
    def compute_node_properties_by_graph(self,node_functions,
                                        verbose = False):
        flu.compute_node_properties_by_graph(self.G,node_functions,
                                            verbose=verbose)
    
    def compute_global_functions(
        self,
        attributes_list = None,
        attributes_list_skip = None,
        global_functions = None,
        verbose = False,):
        
        flu.compute_global_functions(self.G,
        attributes_list = attributes_list,
        attributes_list_skip = attributes_list_skip,
        global_functions = global_functions,
        verbose = verbose,
        )
        
    def subgraph_from_node_query(self,query,node_functions=None,
                  object_argument_name=None,
                               verbose=False,
                                ):
        return flu.subgraph_from_node_query(self.G,
                                            query=query,
                                     node_functions=node_functions,
                                      object_argument_name=object_argument_name,
                                     default_arguments=self.function_args,
                                       verbose=verbose)
        
    def objects_from_node_query(self,query,
                               node_functions=None,
                  object_argument_name=None,
                               verbose=False):
        sub_G = self.subgraph_from_node_query(query,
                                    node_functions=node_functions,
                                    object_argument_name=object_argument_name,
                                    verbose = verbose
                                   )
        return flu.objects_from_G(sub_G,self.attribute_name)
    
    def replace_G(self,G):
        self.G = G
        self._objects = self.objects
    
def objects_from_G(G,attribute_name=default_attribute_name):
    return [G.nodes[k][attribute_name] for k in G.nodes()]
        
def objects_from_node_query(G,query,
                            attribute_name=default_attribute_name,
                            default_arguments=None,
                            node_functions=None,
                          object_argument_name=None,
                               verbose=False):
    """
    Will filter objects stored in a graph based on the node query
    """
    sub_G = flu.subgraph_from_node_query(G,query=query,
                               node_functions=node_functions,
                        object_argument_name=object_argument_name,
                        default_arguments=default_arguments,
                       verbose = verbose)
    return [sub_G.nodes[k][attribute_name] for k in sub_G.nodes()]
    
    

#from python_tools import general_utils as gu
def G_from_objects(objects,
                  attribute_name=default_attribute_name,
                   object_attributes = None,
                  add_fully_connected_edges = True):
    """
    Purpose: to store an object in the nodes of a 
    graph
    """
    node_ids = np.arange(len(objects))
    G = xu.complete_graph_from_node_ids(node_ids)
    
    data_matrix = dict([(k,{attribute_name:v}) if type(v) != dict 
                        else (k,gu.merge_dicts([v,{attribute_name:v}])) for k,v in zip(node_ids,objects)])

    if object_attributes is not None:
        if len(object_attributes) != len(node_ids):
            raise Exception("Object attributes needs to be same list")
        for i,att_dict in enumerate(object_attributes):
            data_matrix[i].update(att_dict)
    
    xu.set_node_attributes_dict(G,data_matrix)
    
    return G


def compute_node_property_with_obj(func,
                          G=None,
                          node=None,
                          obj = None,
                          func_name=None,
                          func_kwargs = None,
                          object_argument_name=None,
                         verbose = False):
    """
    Will run a function for a certain object
    
    Ex: 
    def n_branches(candidate,adder=10):
        return len(candidate['branches']) + adder

    flu.compute_node_property_with_obj(G,0,n_branches,object_argument_name="candidate",
                              func_kwargs = dict(adder=20),
                              verbose = True)
    """
    if obj is None:
        obj = G.nodes[node][default_attribute_name]
    if func_name is None:
        func_name = func.__name__
    if func_kwargs is None:
        func_kwargs = dict()
    
    if object_argument_name is None:
        func_value = func(obj,**func_kwargs)
    else:
        func_kwargs[object_argument_name] = obj
        func_value = func(**func_kwargs)
        
    if verbose:
        print(f"{func_name} for node {node}= {func_value}")
    
    if G is not None and node is not None:
        G.nodes[node][func_name] = func_value
    return func_value
    
    
#from python_tools import general_utils as gu

def compute_node_properties(G,node_functions,
                            object_argument_name=None,
                            default_arguments=None, #global arguments 
                            include_node_stored_args = True,
                            func_mod = None,
                           verbose = False):
    """
    Purpose: will compute 
    multiple properties of the nodes in the graph
    and store them as node properties
    
    Pseudocode: 
    1) 
    """
    G_nodes = list(G.nodes())
    att_dict = dict([(k,dict()) for k in G_nodes])
    if default_arguments is None:
        default_arguments = dict()
    obj_arg_name = object_argument_name
    
    for f_info  in node_functions:
        nodes_to_compute = G_nodes
        default = None
        args = default_arguments.copy()
        if gu.is_function(f_info):
            func = f_info
            f_name = f_info.__name__
        elif type(f_info) == "str": #so can just give function name from mod
            if func_mod is None:
                from neurd import neuron_statistics as nst
                func_mod = nst
            func = geattr(func_mod,f_info)
            f_name = f_info
        elif type(f_info) == dict:
            
            func = f_info["function"]
            f_name = f_info.get("name",func.__name__)
            new_args = f_info.get("arguments",dict()) #function level arguments
            args.update(new_args)
            nodes_to_compute = f_info.get("nodes_to_compute",nodes_to_compute)
            if not nu.is_array_like(nodes_to_compute):
                nodes_to_compute = [nodes_to_compute]
            obj_arg_name = f_info.get("object_argument_name",object_argument_name) 
                
            default = f_info.get("default",default)
        else:
            raise Exception(f"Unimplemented type {type(f_info)}")

        if verbose:
            print(f"Working on {f_name} with args = {args}")
            
        nodes_not_compute = np.setdiff1d(G_nodes,nodes_to_compute)
            
        for n in nodes_to_compute:
            #doing the local update of the arguments that are stored in the node
            if include_node_stored_args:
                curr_args = gu.merge_dicts([args,G.nodes[n].get(node_arguments_name,dict())])
            else:
                curr_args = args
            att_dict[n][f_name] = flu.compute_node_property_with_obj(func,
                          G=G,
                          node=n,
                          obj = None,
                          func_name=f_name,
                          func_kwargs = curr_args,
                          object_argument_name=obj_arg_name,
                         verbose = verbose)
            
        if default is not None:
            for n in nodes_not_compute:
                att_dict[n][f_name] = default
        
    xu.set_node_attributes_dict(G,att_dict)
    return G
    
def subgraph_from_node_query(G,query,
                             node_functions=None,
                  object_argument_name=None,
                             default_arguments=None,
                               verbose=False):
    if node_functions is not None:
        flu.compute_node_properties(G,node_functions,
                        object_argument_name=object_argument_name,
                        default_arguments=default_arguments,
                       verbose = verbose)

    return xu.subgraph_from_node_query(G,query=query)


    
    
def compute_edge_attributes_locally(G,edge_functions):
    """
    If want to cnsider every node in the graph then make
    sure tha graph is fully connected
    """
    pass

# ------ functions computed after object is abstracted away using graph -----#
def compute_node_properties_by_graph(G,
                                 node_functions,
                                  default_arguments=None,
                                    verbose = False,
                                    **kwargs):
    """
    Purpose: Will compute attributes of nodes
    that are only based on the node and the values stored in them
    
    Pseudocode: 
    1) Iterate through the node functions:
        for each extract out the metadata
        2) Iterate through all the nodes
        Compute the function and store as a node property
    """
    G_nodes = list(G.nodes())
    att_dict = dict([(k,dict()) for k in G_nodes])
    
    if default_arguments is None:
        default_arguments = dict()
    
    for f_info  in node_functions:
        nodes_to_compute = G_nodes
        default = None
        args = default_arguments.copy()
        
        if gu.is_function(f_info):
            func = f_info
            f_name = f_info.__name__
        elif type(f_info) == dict:
            
            func = f_info["function"]
            f_name = f_info.get("name",func.__name__)
            new_args = f_info.get("arguments",dict()) #function level arguments
            args.update(new_args)
            nodes_to_compute = f_info.get("nodes_to_compute",nodes_to_compute)
            if not nu.is_array_like(nodes_to_compute):
                nodes_to_compute = [nodes_to_compute]
            default = f_info.get("default",default)
        else:
            raise Exception(f"Unimplemented type {type(f_info)}")

        if verbose:
            print(f"Working on {f_name} with args = {args}")
            
        nodes_not_compute = np.setdiff1d(G_nodes,nodes_to_compute)
            
        for n in nodes_to_compute:
            curr_args = args
            att_dict[n][f_name] = func(
                          G=G,
                          node=n,
                          **curr_args,)
            
        if default is not None:
            for n in nodes_not_compute:
                att_dict[n][f_name] = default
                
    xu.set_node_attributes_dict(G,att_dict)
    return G

def diff_from_max(G,node,attribute_name):
    all_vals = [G.nodes[k][attribute_name] for k in G.nodes()]
    max_val = np.max(all_vals)
    curr_val = G.nodes[node][attribute_name]
    return curr_val - max_val

def diff_from_min(G,node,attribute_name):
    all_vals = [G.nodes[k][attribute_name] for k in G.nodes()]
    min_val = np.min(all_vals)
    curr_val = G.nodes[node][attribute_name]
    return min_val - curr_val

def distance_above_soma_diff_from_max(G,node):
    return diff_from_max(G,node,attribute_name="distance_above_soma")

#from python_tools import numpy_utils as nu
#from . import numpy_dep as np

def compute_global_functions(
    G,
    attributes_list = None,
    attributes_list_skip = None,
    global_functions = None,
    verbose = False,
    ):

    """
    Purpose: To compute the difference between the min 
    and the max between diff attributes 

    """
    if global_functions is None:
        global_functions = (diff_from_max,
                 flu.diff_from_min)

    if attributes_list is None:
        attributes_list = nu.intersect1d_multi_list([list(G.nodes[k].keys()) for k in G.nodes])

    if verbose:
        print(f"attributes_list before skip = {attributes_list}")

    
    if attributes_list_skip is None:
        attributes_list_skip = []
        
    attributes_list_skip += [default_attribute_name]

    attributes_list= np.setdiff1d(attributes_list,attributes_list_skip)
        
    

    if verbose:
        print(f"attributes_list after skip = {attributes_list}")

    for att in attributes_list:
        if verbose:
            print(f"----Working on attribute {att}---")
        for f in global_functions:
            if type(f) != dict:
                f_name = f.__name__
                f_func = f
            else:
                f_name = f["name"]
                f_func = f["function"]


            for n in G.nodes():
                try:
                    curr_value = f_func(G,n,att)
                except:
                    if verbose:
                        print(f"Couldn't compute global functions for att {att}")
                    continue
                G.nodes[n][f"{att}_{f_name}"] = curr_value
                if verbose:
                    print(f"For node {n}, att = {att}, function {f_name}: {curr_value}")

    return G

# ------------- Examples of how to use the filtering ------------- #

#from python_tools import filtering_utils as flu
def filter_candidates_by_query(neuron_obj,
                              candidates,
                              functions_list,
                              query,
                               functions_list_graph=None, #attriutes computed on just the functions
                               return_df_before_query = False,
                               verbose = False,
                              **kwargs):
    """
    Purpose: To use the filtering module
    to filter the candidates according to functions
    and the query listed
    
    Ex: 
    min_skeletal_length = 150000
    min_distance_above_soma = 100000

    objs_after_query,node_df  = flu.filter_candidates_by_query(neuron_obj,
                                  candidates = apical_candidates,
                                  functions_list=[dict(function=nst.skeletal_length_over_candidate,
                                                       name="skeletal_length"),
                                                 dict(function=nst.max_layer_distance_above_soma_over_candidate,
                                                     name="distance_above_soma")],
                                  query=f"(skeletal_length > {min_skeletal_length}) and (distance_above_soma > {min_distance_above_soma})",
                                                          return_df_before_query=True)

    """
    
    """
    Old method: 
    objs = [{flu.default_attribute_name:k,
             flu.node_arguments_name:{"limb_obj":neuron_obj[k["limb_idx"]],
                        "start_node":k["start_node"]}
            }  for k in candidates]
    C = flu.Comparator(objs,function_args=dict(neuron_obj=neuron_obj))
    """
    
    objs_attrs = [{
             flu.node_arguments_name:{"limb_obj":neuron_obj[k["limb_idx"]],
                        "start_node":k["start_node"]}
            }  for k in candidates]
    C = flu.Comparator(candidates,
                       function_args=dict(neuron_obj=neuron_obj),
                      object_attributes = objs_attrs)
    
 
    C.compute_node_properties(functions_list,
                             object_argument_name="candidate")
    
    if functions_list_graph is not None:
        C.compute_node_properties_by_graph(functions_list_graph)
    
    sub_objs = C.objects_from_node_query(query=query)
    if return_df_before_query:
        return sub_objs,C.node_df
    else:
        return sub_objs
    
    
def filter_to_one_by_query(
    Comparator_obj,
    queries,
    functions_list=None,
    functions_list_graph=None, #attriutes computed on just the functions
    attributes_for_global_comparisons = None,
    return_df_before_query = False,
    verbose = False,
    go_to_next_query_if_zero = True,
    ):
    """
    Purpose: to filter a comparator object down
    to one node based on the queries
    
    """
    C = Comparator_obj
    if functions_list is not None:
        C.compute_node_properties(functions_list,
                                 object_argument_name="candidate")
    
    if functions_list_graph is not None:
        C.compute_node_properties_by_graph(functions_list_graph)

    if attributes_for_global_comparisons is not None:
        C.compute_global_functions()

    return_df = C.node_df

    for j,q in enumerate(queries):
        if verbose:
            print(f"Working on query {j} : Starting with {len(C.G.nodes())}")
            print(f"Query = {q}")

        #b) query the graph
        sub_G = C.subgraph_from_node_query(query = q)

        n_nodes_left = len(sub_G.nodes())
        if verbose:
            print(f"n_nodes_left = {n_nodes_left}")

        
        if n_nodes_left == 1:
            if verbose:
                print(f"Only one remaining candidate so breaking")
            C.replace_G(sub_G)
            break
        elif n_nodes_left > 1:
            C.replace_G(sub_G)
            continue
        else:
            if go_to_next_query_if_zero:
                continue
            else:
                C.replace_G(sub_G)
                break

    remaining_objs = C.objects

    if verbose:
        print(f"# of remaining_objs = {len(remaining_objs)}")

    if return_df_before_query:
        return remaining_objs,return_df
    else:
        return remaining_objs

def filter_candidates_to_one_by_query(neuron_obj,
                                     candidates,
                                      queries,
                                      candidates_attributes = None,
                                     functions_list=None,
                                      functions_list_graph=None, #attriutes computed on just the functions
                                      attributes_for_global_comparisons = None,
                                      return_df_before_query = False,
                                      verbose = False
                                     ):
    """
    Purpose: To compute attributes of each 
    candidate and then check if only one candidate
    passes a certain threshold 
    --> if only one does then mark that as the winner

    Psuedocode: 
    0) Define a list of functions and the threshold to filter
    1) compute the features over the candidates
    For each query (--> the order will matter)
    b) query the graph 
    c1) if one then break, 
    c2) if 0, restore the original graph and continue
    c3) if more than 1, use resultant graph and continue

    """
    
    #a) compute the node features
    objs = [{flu.default_attribute_name:k,
             flu.node_arguments_name:{"limb_obj":neuron_obj[k["limb_idx"]],
                        "start_node":k["start_node"]}
            }  for k in candidates]

    C = flu.Comparator(candidates,
                       function_args=dict(neuron_obj=neuron_obj),
                       object_attributes = objs,
                      )
    
    if functions_list is not None:
        C.compute_node_properties(functions_list,
                                 object_argument_name="candidate")
    
    return flu.filter_to_one_by_query(
    Comparator_obj=C,
    queries=queries,
    functions_list=functions_list,
    functions_list_graph=functions_list_graph, #attriutes computed on just the functions
    attributes_for_global_comparisons = attributes_for_global_comparisons,
    return_df_before_query = return_df_before_query,)
    
    
        
    
        
        
#from python_tools import filtering_utils as flu



#--- from python_tools ---
from . import general_utils as gu
from . import networkx_utils as xu
from . import numpy_utils as nu
from . import tqdm_utils as tqu

from . import filtering_utils as flu