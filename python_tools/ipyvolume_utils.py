"""
Purpose: To summarize and expose ipyvolume functionality

Tutorial: https://www.youtube.com/watch?v=hOKa8klJPyo

Notes: 
- whenever changing properties on ipyvolume you do so through the returned object and the transition is interpolated

The upper left controls: 
1) Eye: allows 2 outputs from different angles for VR
2) The icon can select 3D so can output data for 3D viewing in google cardboard
(180 or 360 degrees)
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
    
import numpy_utils as nu
def example_plot_line_segments(array = None):
    if array is None:
        array = np.array([[[0,0,0],[1,1,1]],
                  [[1,1,1],[2,0,-3]]])
        
    ipv.figure()
    vertices,lines = nu.coordinates_edges_from_line_segments(
        array
    )

    mesh2 = ipv.plot_trisurf(vertices[:,0], 
                                vertices[:,1], 
                                vertices[:,2], 
                                lines=lines)
    ipv.show()
    return mesh2

example_plot_skeleton = example_plot_line_segments
    
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
    
def example_plot_line():
    fig = ipv.figure()
    u = np.linspace(0,1,4000)
    r = 1 + 0.3*np.sin(u*np.pi*4)
    x = np.sin(u*2*np.pi*40) *r
    y = np.cos(u*2*np.pi*40) *r
    z = u
    line = ipv.plot(x,y,z)
    fig.camera.rotateY(1)
    ipv.show()

    change_line = widgets.Button(description="change line")

    def change_line_func(info):
        r = 1 + 1*np.sin(u*np.pi*3)
        line.x = np.sin(u*2*np.pi*40) *r
        line.y = np.cos(u*2*np.pi*40) *r
        line.color = np.stack([u*0,u,u*0],1)

    change_line.on_click(change_line_func)
    display(change_line)
    
def animation_off():
    fig = ipv.gcf()
    fig.animation = 0
    
    
def example_scatter(N=1000):
    x,y,z = np.random.normal(0,1,(3,N))
    return x,y,z
def example_selection():
    x,y,z = example_scatter()
    fig = ipv.figure()
    scatter = ipv.scatter(x,y,z,marker = "sphere",color = "green")
    ipv.selector_default()
    out = widgets.Output()
    ipv.show()

    @out.capture(clear_output=True,wait = True)
    def print_info(*_):
        indices = scatter.selected[0]
        meanx = np.mean(scatter.x[indices])
        print(f"mean of x = {meanx}")

    display(out)
    scatter.observe(print_info,"selected")
    
def set_style(style="light"):
    """
    Prospective styles: light, dark
    """
    ipv.style.use(style)
    
"""
Animations: if feed into a variable like x or y
a 2D array, knows that the first component is time
and then can animate by using the sequence_index

ex: x.shape = (200,313) # 200 time points, 313 data

"""

def set_selector_default():
    ipv.selector_default()

def add_animation_widget(obj,interval = 400,**kwargs):
    """
    interval: interval in msec between each frame
    """
    ipv.animation_control(obj,interval=interval,**kwargs)

    
def example_time_quiver_dataset():
    data = ipv.datasets.animated_stream.fetch().data[...,::4]
    x,y,z,vx,vy,vz = data
    return x,y,z,vx,vy,vz 

def change_time_interval(obj,time_idx=4):
    obj.sequence_index = time_idx
    
def example_animation_through_time():
    """
    Psueodocode: 
    1) Create a time dataset
    2) Create a quiver using the 2D dataset
    3) Add animation control
    4) Add widget to control the properties
    
    
    """
    x,y,z,vx,vy,vz  = example_time_quiver_dataset()
    
    # 2) Creating the quiver
    fig = ipv.figure()
    ipv.style.use("dark")
    quiver = ipv.quiver(x,y,z,vx,vy,vz,size = 5,)
    ipv.show()
    
    #3) Adding the Animation Control
    ipv.animation_control(quiver,interval = 400)
    
    
    #4) Adding the control on widget properties
    w = widgets.ToggleButtons(
    options=['arrow','sphere','cat'],
    value = "sphere")

    widgets.link((quiver,"geo"),(w,"value"))
    display(w)
    
    
"""
Making movies: 
1) Add: adds keyframes
2) Set the interpolation to smooth
3) Press play


"""
from ipyvolume.moviemaker import MovieMaker
def example_movie():
    ipvu.example_animation_through_time()
    fig = ipv.gcf()
    
    mm = MovieMaker(stream = fig,camera = fig.camera)
    display(mm.widget_main)
    
import ipywebrtc as webrtc
def example_movie_recorded(filename="bc_example"):
    """
    Hit the begin record button, then stop the record button
    and then it will play back the saved video
    """
    ipvu.example_movie()
    fig = ipv.gcf()
    display(webrtc.VideoRecorder(stream=fig,filename=filename))
    
import ipyvolume_utils as ipvu
    
    
    
    