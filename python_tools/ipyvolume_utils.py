"""
Purpose: To summarize and expose ipyvolume functionality

Notes: 
- whenever changing properties on ipyvolume you do so through the returned object and the transition is interpolated
"""

import ipyvolume as ipv
import ipywidgets as widgets
from IPython.display import display
import numpy as np

def example_widgets_linked_to_attributes():
    """
    Pseudocode: 
    1) Create dummy data
    2) Create scatter plot and store result
    3) Link the results attributes to different widgets
    
    """
    N = 1000
    x,y,z = np.random.normal(0,1,(3,N))
    
    fig = ipv.figure()
    scatter = ipv.scatter(x,y,z,marker = "sphere")
    ipv.show()
    
    w = widgets.ToggleButtons(options = ["sphere","box","diamond","circle_2d","point_2d","arow"],
                             description="Scatter Marker")
    widgets.link((scatter,"geo"),(w,"value"))
    
    slider = widgets.FloatSlider(min = 0.0, max = 10, step = 0.1,
                                description="Scatter Size")
    widgets.link((scatter,"size"),(slider,"value"))
    
    display(w,slider)
    
    
def example_quiver_plot_and_attributes():
    N = 1000
    x,y,z = np.random.normal(0,1,(3,N))

    fig = ipv.figure()
    quiver = ipv.quiver(x,y,z,x,y,z)

    ipv.show()
    
    # --- Adds a button for flipping ---
    flip_button = widgets.Button(
        description = ("Flip Arrows")
    )

    def flip_arrows_func(key):
        quiver.vx = -quiver.vx
        quiver.vy = quiver.vy
        quiver.vz = -quiver.vz

    flip_button.on_click(flip_arrows_func)
    display(flip_button)
    
    cp = widgets.ColorPicker(description = "Pick Color")
    widgets.jsdlink((cp,"value"),(quiver,"color"))
    display(cp)
    
def set_xyzlim(axes_min,axes_max):
    ipv.xyzlim(axes_min,axes_max)
    
s = 1/2**0.5
tetrahedron_vertices = np.array([[1,0,-s],
             [-1,0,-s],
             [0,1,s],
             [0,-1,s]])
tetrahedron_triangles = triangles = np.array([[0,1,2],
                     [0,1,3],
                     [0,2,3],
                     [1,3,2]])
    
def example_mesh():
    x,y,z = tetrahedron_vertices.T

    ipv.figure()
    ipv.plot_trisurf(x,y,z,triangles = tetrahedron_triangles,color = "orange")
    ipv.scatter(x,y,z,marker="sphere",color = "blue")
    ipv.xyzlim(-2,2)
    ipv.show()
    
    
def example_surface_not_mesh():
    a = np.linspace(-5,5,30)
    X,Y = np.meshgrid(a,a)

    Z = X*Y**2

    ipv.figure()
    mesh = ipv.plot_surface(X,Y,Z, color = "orange")
    ipv.show()

    alter_button = widgets.Button(description="alter_surface")

    def alter_func(data):
        mesh.y = -mesh.y
        mesh.x = mesh.x + 1.4
        mesh.color = "green"

    alter_button.on_click(alter_func)
    display(alter_button)