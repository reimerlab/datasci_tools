"""
Installation notes: 
grandiso only installs 1.0.0 (and not 2.0.0) in python3.7 and not python3.6 (checked with Docker containers)

Node and edge constraints: https://github.com/aplbrain/dotmotif/wiki/Attributes

Ex of Edge constraint
A -> B [weight <= 20, weight >= 10, weight != 12]

Node constraint: 
A -> B [weight >= 0.6]
A.type = "inh" 
A.layer in [5,4]
B.type = "exc"
B.layer = 1

Graph has to be a DiGraph and use the NetworkXExecutor as
the searcher

Ex: 

from dotmotif import Motif, GrandIsoExecutor, NetworkXExecutor

import networkx as nx
G = nx.Graph()
G.add_edges_from([[1,2],[2,3]])
G = nx.DiGraph(G)
nx.draw(G,with_labels=True)

motif = Motif('''
A -> B
B -> C
''')

E = NetworkXExecutor(graph=G)

results = E.find(motif)
print(len(results))


Example Motifs: 
motif = Motif('''
u -> d1
d1 -> d2
u.flavor =  "banana"
''')



Helpful tips: 

1) negating connections
person2 !> person1 # means the connection does not exist

2) How to do an or operator
motif = '''
A -> B 
B -> A

A.cell_type = "excitatory"
B.cell_type in "inhibitoryexcitatory"

3) not in or not contains
"!in": 
"!contains"

'''

"""
try:
    import dotmotif
except ImportError as e:
    raise e("DotMotif must be installed from https://github.com/bacelii/dotmotif/")
    
from dotmotif import Motif, NetworkXExecutor
try:
    from dotmotif import GrandIsoExecutor
except:
    pass 

import dotmotif 
# if dotmotif.__version__ != '0.9.2b':
#     raise Exception("")
import networkx as nx
from . import networkx_utils as xu


import copy

def cancel_characters_attribute(
    s,
    attribute_characters_to_cancel = None,
    ):
    import re

    if attribute_characters_to_cancel is None:
        attribute_characters_to_cancel = ("-",)
        
    for k in attribute_characters_to_cancel:
        pattern = (
            f"([a-zA-Z0-9]){k}([a-zA-Z0-9])")
        s = re.sub(pattern,r"\1\2",s)
        
    return s

def convert_characters_for_search(
    G,
    motif,
    attributes_to_replace_numbers=None,
    attribute_characters_to_cancel = None,
    verbose = False
    ):
    from . import regex_utils as ru
    G = copy.deepcopy(G)
    
    number_dict = {
       "1":"one",
       "2":"two",
       "3":"three",
       "4":"four",
       "5":"five",
       "6":"six",
       "7":"seven",
       "8":"eight",
       "9":"nine",
       "0":"zero", 
    }

    

    
    motif = ru.multiple_replace(motif,number_dict)
    motif = dmu.cancel_characters_attribute(
        motif,
        attribute_characters_to_cancel=attribute_characters_to_cancel,
        
    )
        
    for n in G.nodes():
        for k in G.nodes[n]:
            if type(G.nodes[n][k]) != str:
                continue
            if attributes_to_replace_numbers is not None:
                if k not in attributes_to_replace_numbers:
                    continue

            G.nodes[n][k] = ru.multiple_replace(
                G.nodes[n][k],
                number_dict
            )
            
            G.nodes[n][k] = dmu.cancel_characters_attribute(
                G.nodes[n][k],
                attribute_characters_to_cancel=attribute_characters_to_cancel,
            )

    for e1,e2 in G.edges():
        for k in G[e1][e2]:
            if type(G[e1][e2]) != str:
                continue
            if attributes_to_replace_numbers is not None:
                if k not in attributes_to_replace_numbers:
                    continue

            G[e1][e2][k] = ru.multiple_replace(
                G[e1][e2][k],
                number_dict
            )
            
            G[e1][e2][k] = dmu.cancel_characters_attribute(
                G[e1][e2][k],
                attribute_characters_to_cancel=attribute_characters_to_cancel,
            )
    # convert the dashes
    if verbose:
        print(f"motif= {motif}")
    return G,motif
    

def graph_matches(
    G,
    motif,
    verbose = False,
    Executor = None,
    convert_characters = False,
    attributes_to_replace_numbers=None,):
    """
    Purpose: to find the number of subgraphs of 
    an existing motif in a graph structure; 
    
    Ex: 
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([[1,2],[2,3]])
    G = nx.DiGraph(G)
    nx.draw(G,with_labels=True)
    xu.set_node_attributes_dict(G,{2:dict(flavor="apple"),3:dict(flavor = "banana")})

    import dotmotif_utils as dmu
    dmu.n_graph_matches(G,motif)
    
    """
    
    graph = xu.convert_to_non_multi(G)
    if convert_characters:
        
        graph,motif = dmu.convert_characters_for_search(
                graph,
                motif,
                attributes_to_replace_numbers=attributes_to_replace_numbers,
            )
        #print(f"motif = {motif}")
    motif = Motif(motif)
    

    
    if Executor is None:
        E = NetworkXExecutor(graph=graph)
    else:
        E = Executor
    results = E.find(motif)
    if verbose:
        print(f"# of matches = {len(results)}")
    
    return results

import time
def graph_search(motif,Executor = None,
                 G = None,
                 enforce_inequality = False,
                 exclude_automorphisms = False,
                 verbose = False,
                 
                 ):
    """
    Purpose: to find the number of subgraphs of 
    an existing motif in a graph structure; 
    
    Arguments: 
    enforce_inequality: if nodes in with different names in the query can potentially be the same node
    A --> B --> C ( A != C if enforce_inequality == True)
    
    Ex: 
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([[1,2],[2,3]])
    G = nx.DiGraph(G)
    nx.draw(G,with_labels=True)
    xu.set_node_attributes_dict(G,{2:dict(flavor="apple"),3:dict(flavor = "banana")})

    import dotmotif_utils as dmu
    dmu.graph_search(G,motif)
    
    
    
    """
    st = time.time()
    motif = Motif(motif,
                #enforce_inequality = enforce_inequality,
                #exclude_automorphisms = exclude_automorphisms,
                 )
    
    if Executor is None:
        E = NetworkXExecutor(graph=G)
    else:
        E = Executor#(graph=G)
        if verbose:
            print(f"type(Excecutor) = {type(E)}")
    results = E.find(motif)
    if verbose:
        print(f"# of matches = {len(results)}")
        print(f"total time: {time.time() - st}")
    
    return results

def n_graph_matches(G,motif,verbose = False,**kwargs):
    return len(graph_matches(G,motif,verbose,**kwargs))

# ----------------- A Lot of different motif configs --------

'''
exp = """\
macro(A) {
    A.type = "excitatory"
    A.size >= 4.0
}
Aaa -> Ba
macro(Aaa)
macro(Ba)
"""

# dynamic constraint where can compare values
exp = """\
macro(A, B) {
    A.radius > B.radius
}
macro(A, B)
A -> B
"""

exp = """\
dualedge(A, B) {
A -> B
B -> A
}
dualedge(C, D)
"""

exp = """\
tri(A, B, C) {
    A -> B
    B -> C
    C -> A
}
tri(C, D, E)
"""

#nested macros
exp = """\
dualedge(A, B) {
    A -> B
    B -> A
}
dualtri(A, B, C) {
    dualedge(A, B)
    dualedge(B, C)
    dualedge(C, A)
}
dualtri(foo, bar, baz)
"""


# how to do comments in macros: 

exp = """\
# Outside comment
edge(A, B) {
    # Inside comment
    A -> B
}
dualedge(A, B) {
    # Nested-inside comment
    edge(A, B)
    edge(B, A)
}
dualedge(foo, bar)
"""


# macro with edge attributes
exp = """\
macro(Aa, Ba) {
    Aa -> Ba [type != 1, type != 12]
}
macro(X, Y)
"""

'''

from . import dotmotif_utils as dmu


