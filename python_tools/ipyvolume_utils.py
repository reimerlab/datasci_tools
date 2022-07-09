"""
Purpose: To summarize and expose ipyvolume functionality

Tutorial: https://www.youtube.com/watch?v=hOKa8klJPyo
Documentation: https://ipyvolume.readthedocs.io/_/downloads/en/docs/pdf/

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

def print_selection_hotkeys():
    s = """
    Now hold the control key to do selections, type

    ‘C’ for circle
    ‘R’ for rectangle
    ‘L’ for lasso
    ‘=’ for replace mode
    ‘&’ for logically and mode
    ‘|’ for logically or mode
    ‘-’ for subtract mode
    """
    print(s)

def example_scatter():
    N = 1000
    x,y,z = np.random.normal(0,1,(3,N))
    
    fig = ipv.figure()
    scatter = ipv.scatter(x,y,z,marker = "sphere")
    ipv.show()
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
    
    
def example_quiver_plot_and_attributes(offset = 0):
    N = 1000
    x,y,z = np.random.normal(0,1,(3,N)) + offset

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
    
    
"""
For selection:
Can add the kwargs: color_selected to control color of the selected


"""
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
    
# ------------- utils to help with plotting ---- 
import ipywidgets as widgets
def add_attribute_widget(
    obj,
    widget,
    attribute,
    description=None,
    description_prefix = None,
    default_value = None,
    display_widget = False,
    **kwargs):
    
    if description is None:
        description = attribute.title()
        
    if description_prefix is not None:
        description = f"{description_prefix} {description}"
    
    if default_value is not None:
        value = default_value
    else:
        value = None
        
    try:
        w = getattr(widgets,widget)(
            description=description,
            value=value,
            **kwargs)
    except:
        w = getattr(widgets,widget)(
            description=description,
            **kwargs)
    
    if widget in ["ColorPicker"]:
        link_func = "jslink"
    else:
        link_func = "link"
    curr_link = getattr(widgets,link_func)((obj,attribute),(w,"value"))
    
    if display_widget:
        display(w)
    return w

def add_size_widget(
    obj,
    min=0,
    max=3,
    **kwargs
    ):
    
    return ipvu.add_attribute_widget(
        obj,
        widget="FloatSlider",
        attribute = "size",
        min=min,
        max=max,
        **kwargs)

def add_color_widget(
    obj,
    **kwargs
    ):
    
    return ipvu.add_attribute_widget(
        obj,
        widget="ColorPicker",
        attribute = "color",
        **kwargs)

marker_options= (
    "arrow","box","diamond","sphere","point_2d","square_2d","triangle_2d","circle_2d","cat",
)

def add_marker_widget(
    obj,
    options=marker_options,
    default_value = "sphere",
    **kwargs
    ):
    
    return ipvu.add_attribute_widget(
        obj,
        widget="Dropdown",
        attribute = "geo",
        default_value=default_value,
        options=options,
        **kwargs)

def set_axes_lim(
    min=None,
    max=None,
    lim = None,
    xlim = None,
    ylim = None,
    zlim = None,
    ):
    
    if min is not None and max is not None:
        ipv.xlim(min[0],max[0])
        ipv.ylim(min[1],max[1])
        ipv.zlim(min[2],max[2])
        return
        
    if xlim is None:
        xlim = lim
        
    if ylim is None:
        ylim = lim
        
    if zlim is None:
        zlim = lim
        
    ipv.xlim(xlim[0],xlim[1])
    ipv.ylim(ylim[0],ylim[1])
    ipv.zlim(zlim[0],zlim[1])
    
def set_axes_visibility(visibility=True):
    if not visibility:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
        
def save_to_html(path):
    ipv.pylab.save(path)
    
def set_zoom(
    center_coordinate,
    radius=0,
    radius_xyz = None,
    show_at_end=False,
    flip_y = True,
    axis_visibility=False):
    
    
    coord = np.array(center_coordinate)
    
    if flip_y:
        coord[...,1] = -coord[...,1]
        
    if radius_xyz is None:
        radius_xyz = np.array([radius,radius,radius])
        
    coord_radius = [k if k is not None else radius for k in radius_xyz]
    ipv_function = [ipv.xlim,ipv.ylim,ipv.zlim]
    for c,c_rad,ipvf in zip(coord,coord_radius,ipv_function):
        ipvf(c - c_rad, c + c_rad)

    ipvu.set_axes_visibility(axis_visibility)
    if show_at_end:
        ipv.show()  
        
def clear_figure():
    ipv.clear()
    
def get_all_fig_scatters(verbose = False):
    fig = ipv.gcf()
    return_list = fig.scatters
    if verbose:
        print(f"# of scatter objects = {len(return_list)}")
    return return_list

def get_all_fig_meshes(verbose = False):
    fig = ipv.gcf()
    return_list = fig.meshes
    if verbose:
        print(f"# of meshes objects = {len(return_list)}")
    return return_list

def get_all_fig_objs(verbose=False):
    """
    Purpose: To get all the objects
    of the current figure
    """
    fig = ipv.gcf()
    return_list = (
        ipvu.get_all_fig_scatters(verbose=verbose) 
        + ipvu.get_all_fig_meshes(verbose=verbose) 
    )
    
    return return_list

def coords_from_obj(obj):
    """
    Purpose: Find the coordinates of an obj
    """
    if len(obj.x) == 0:
        return np.array([]).reshape(-1,3)
    return np.vstack([getattr(obj,k) for k in ["x","y","z"]]).T.astype("float")

def coords_min_max_from_obj(obj):
    coords = coords_from_obj(obj)
    if len(coords) > 0:
        return np.vstack([np.min(coords,axis=0),np.max(coords,axis=0)])
    else:
        return np.array([]).reshape(-1,3)
    
def coords_min_max_from_fig(
    verbose = False,
    ):
    """
    Purpose: to find the min and max coordinates
    of all the objects in a current figure
    """

    all_objs = ipvu.get_all_fig_objs(verbose=verbose)

    min_coords = []
    max_coords = []

    for k in all_objs:
        coords = ipvu.coords_min_max_from_obj(k)
        if len(coords) > 0:
            min_coords.append(coords[0])
            max_coords.append(coords[1])

    if len(min_coords) > 0:
        all_min = np.min(np.array(min_coords).reshape(-1,3),axis=0)
        all_max = np.max(np.array(max_coords).reshape(-1,3),axis=0)
        return_coords =  np.vstack([all_min,all_max])
        if verbose:
            print(f"Min,Max = {return_coords}")
        return return_coords
        
    else:
        return np.array([]).reshape(-1,3)

def set_axes_lim_from_fig(
    buffer = 0,
    verbose = False,
    ):
    """
    Purpose: To automatically set the zoom to include
    all the coordinates of the current scatters
    and meshes
    """

    

    min_max_coords = ipvu.coords_min_max_from_fig(verbose = verbose) + buffer
    if len(min_max_coords) > 0:
        ipvu.set_axes_lim(min=min_max_coords[0],max=min_max_coords[1])
    
def scatter_plot_func(
    array,
    size = 0.3,
    marker = "sphere",
    **kwargs
    ):
    return ipv.scatter(*ipvu.xyz_from_array(array),
                       size=size,marker=marker
                      )
    
def surface_plot_func(
    array,
    **kwargs
    ):
    return ipv.plot_surface(*ipvu.xyz_from_array(array))
    
def line_plot_func(
    array,
    **kwargs
    ):
    return ipv.plot(*ipvu.xyz_from_array(array))

def line_segments_plot_func(
    array,
    lines=None,
    **kwargs):
    
    if lines is None:
        vertices,lines = nu.coordinates_edges_from_line_segments(
            array
        )
    else:
        vertices = array
    
    return ipv.plot_trisurf(vertices[:,0], 
                                vertices[:,1], 
                                vertices[:,2], 
                                lines=lines)
    
def trisurf_plot_func(
    array,
    triangles,
    **kwargs
    ):
    x,y,z = ipvu.xyz_from_array(array)
    return ipv.plot_trisurf(x,y,z,
        triangles = triangles,
                           )
    
    
def plot_obj(
    array,
    plot_type="scatter",
    #all possible inputs to functions
    triangles = None,
    lines=None,
    color = "green",
    widgets_to_plot = ("size","marker","color"),
    show_at_end = True,
    new_figure = True,
    flip_y = False,
    **kwargs,
    ):
    """
    Purpose: To plot an object
    using a plot type (and possibly add on widgets)
    to help control
    
    import ipyvolume_utils as ipvu
    import ipyvolume as ipv

    ipvu.plot_obj(
        array = cell_centers,
        flip_y = True
    )
    """
    array = np.array(array).astype("float")
    if flip_y:
        array[...,1] = -array[...,1]

    if new_figure:
        ipv.figure()
        
    scat = getattr(ipvu,f"{plot_type}_plot_func")(
        array,
        triangles = triangles,
        lines=lines,
    )
    
    scat.color = color
    

    widget_list = []
    
    if widgets_to_plot is None:
        widgets_to_plot = []
    for w in widgets_to_plot:
        curr_widget = getattr(ipvu,f"add_{w}_widget")(scat,prefix=w,**kwargs)
        widget_list.append(curr_widget)
        
    if len(widget_list) > 0:
        display(widgets.HBox(widget_list))
        
    ipvu.set_axes_lim_from_fig()

    if show_at_end:
        ipv.show()
    
#def add_marker_selection()
def xyz_from_array(array):
    return array[:,0],array[:,1],array[:,2]
    
import ipyvolume_utils as ipvu
    
    
    
    