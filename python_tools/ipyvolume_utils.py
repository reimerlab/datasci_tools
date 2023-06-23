'''

Purpose: To summarize and expose ipyvolume functionality

Tutorial: https://www.youtube.com/watch?v=hOKa8klJPyo
Documentation: https://ipyvolume.readthedocs.io/_/downloads/en/docs/pdf/

Notes: 
- whenever changing properties on ipyvolume you do so through the returned object and the transition is interpolated
- ARRAYS HAVE TO BE FLOATS TO SHOW UP

The upper left controls: 
1) Eye: allows 2 outputs from different angles for VR
2) The icon can select 3D so can output data for 3D viewing in google cardboard
(180 or 360 degrees)

'''
from IPython.display import display
from ipyvolume.moviemaker import MovieMaker
import ipyvolume as ipv
import ipywebrtc as webrtc
import ipywidgets
import ipywidgets as widgets
import numpy as np
import os

#import ipyvolume as ipv
#import ipywidgets as widgets
#from IPython.display import display
#import numpy as np

def print_selection_hotkeys():
    s = """
    Now hold the CTRL key to do selections, type

    ‘C’ for circle
    ‘R’ for rectangle
    ‘L’ for lasso
    ‘=’ for replace mode
    ‘&’ for logically and mode
    ‘|’ for logically or mode
    ‘-’ for subtract mode
    """
    print(s)

def example_scatter_plot():
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
    new_mesh = ipv.plot_trisurf(x,y,z,triangles = tetrahedron_triangles,color = "orange")
    ipv.scatter(x,y,z,marker="sphere",color = "blue")
    ipv.xyzlim(-2,2)
    ipv.show()
    return new_mesh
    
#from python_tools from . import numpy_utils as nu
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
    ipvu.print_selection_hotkeys()
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
    
def scatter_selected_idx(scatter):
    if scatter.selected is None:
        return []
    return scatter.selected[0]
    
def set_style(style="light"):
    """
    Prospective styles: light, dark
    """
    ipv.style.use(style)
    
def set_dark_background():
    ipv.style.use("dark")
    
def set_white_background():
    ipv.style.use("light")
    
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
#from ipyvolume.moviemaker import MovieMaker
def example_movie():
    ipvu.example_animation_through_time()
    fig = ipv.gcf()
    
    mm = MovieMaker(stream = fig,camera = fig.camera)
    display(mm.widget_main)
    
#import ipywebrtc as webrtc
def example_movie_recorded(filename="bc_example"):
    """
    Hit the begin record button, then stop the record button
    and then it will play back the saved video
    """
    ipvu.example_movie()
    fig = ipv.gcf()
    display(webrtc.VideoRecorder(stream=fig,filename=filename))
    
# ------------- utils to help with plotting ---- 
#import ipywidgets as widgets
def add_attribute_widget(
    obj,
    widget,
    attribute,
    description=None,
    description_prefix = None,
    default_value = None,
    display_widget = False,
    **kwargs):
    
    #print(f"description_prefix = {description_prefix}")
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

def add_alpha_widget(
    obj,
    **kwargs
    ):
    
    return ipvu.add_attribute_widget(
        obj,
        widget="FloatSlider",
        attribute = "alpha",
        min=0,
        max = 1,
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
    debug = False,
    ):
    
    if debug:
        print(f"Start func")
        print(f"xlim = {xlim}")
        print(f"ylim = {ylim}")
        print(f"zlim = {zlim}")
        

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
        
    if debug:
        print(f"Inside func")
        print(f"xlim = {xlim}")
        print(f"ylim = {ylim}")
        print(f"zlim = {zlim}")
    ipv.xlim(xlim[0],xlim[1])
    ipv.ylim(ylim[0],ylim[1])
    ipv.zlim(zlim[0],zlim[1])
    
    set_view(distance = 2)
    
def get_axes_lim(fig = None):
    if fig is None:
        fig = ipv.gcf()
    old_lims = np.vstack([fig.xlim,fig.ylim,fig.zlim]).T
    return old_lims

def set_axes_lim_to_cube(
    bbox = None):
    """
    Purpose: To set the limits of the xyz
    to be a square equal to the longest current size
    """
    if bbox is None:
        bbox = ipvu.get_axes_lim() 
    new_bbox = nu.bbox_cube_from_bbox(
        bbox = bbox,
        verbose = False,
    )

    ipvu.set_axes_lim(
        min = new_bbox[0,:],
        max = new_bbox[1,:],
    )
    
def set_axes_visibility(visibility=True):
    if not visibility:
        ipv.style.axes_off()
        ipv.style.box_off()
    else:
        ipv.style.axes_on()
        ipv.style.box_on()
        
def save_to_html(path):
    ipv.pylab.save(path)
    
def get_xyz_lim():
    fig = ipv.gcf()
    return np.vstack([fig.xlim,fig.ylim,fig.zlim]).T

def get_xyz_radius():
    curr_ax = get_xyz_lim()
    return np.abs(curr_ax[1] - curr_ax[0])/2

def get_center_coordinate(flip_y = True):
    coord = np.mean(get_xyz_lim(),axis=0)
    if flip_y:
        coord[...,1] = -coord[...,1]
    return coord

get_camera_center = get_center_coordinate

def set_zoom(
    center_coordinate=None,
    radius=None,
    radius_xyz = None,
    show_at_end=False,
    flip_y = True,
    axis_visibility=None):
    
    
    if center_coordinate is None:
        center_coordinate = get_center_coordinate(flip_y=flip_y)
        
    coord = np.array(center_coordinate)
    
    if flip_y:
        coord[...,1] = -coord[...,1]
        
    if radius_xyz is None:
        if radius is None:
            radius_xyz = get_xyz_radius()
        else:
            radius_xyz = np.array([radius,radius,radius])
            
        
        
    coord_radius = [k if k is not None else radius for k in radius_xyz]
    ipv_function = [ipv.xlim,ipv.ylim,ipv.zlim]
    for c,c_rad,ipvf in zip(coord,coord_radius,ipv_function):
        ipvf(c - c_rad, c + c_rad)

    if axis_visibility is not None:
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

    
    buffer_array = np.array([[-buffer,-buffer,-buffer,],
                             [buffer,buffer,buffer,]
                            ])
    min_max_coords = ipvu.coords_min_max_from_fig(verbose = verbose) + buffer_array
    if len(min_max_coords) > 0:
        ipvu.set_axes_lim(min=min_max_coords[0],max=min_max_coords[1])
    
def scatter_plot_func(
    array,
    size = 0.3,
    marker = "sphere",
    **kwargs
    ):
    return ipv.scatter(*ipvu.xyz_from_array(array),
                       size=size,
                       marker=marker,
                       **kwargs
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

skeleton_plot_func = line_segments_plot_func
    
def trisurf_plot_func(
    array,
    triangles,
    **kwargs
    ):
    x,y,z = ipvu.xyz_from_array(array)
    return ipv.plot_trisurf(x,y,z,
        triangles = triangles,
                           )

mesh_plot_func = trisurf_plot_func

#from python_tools from . import matplotlib_utils as mu
    
def plot_obj(
    array,
    plot_type="scatter",
    #all possible inputs to functions
    triangles = None,
    lines=None,
    color = "green",
    size = None,
    plot_widgets = True,
    widgets_to_plot = ("size","marker","color"),
    widget_description_prefix = None,
    show_at_end = True,
    new_figure = True,
    flip_y = False,
    axis_visibility = True,
    individually_scale_each_axis = False,
    alpha = None,
    **kwargs,
    ):
    """
    Purpose: To plot an object
    using a plot type (and possibly add on widgets)
    to help control
    
    from python_tools from . import ipyvolume_utils as ipvu
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
        ipv.pylab.clear()
        ipv.clear()
        ipv.figure()
        
        
    if lines is not None:
        plot_type = "line_segments"
    elif triangles is not None:
        plot_type = "mesh"
        
        
    if len(array) > 0:
        scat = getattr(ipvu,f"{plot_type}_plot_func")(
            array,
            triangles = triangles,
            lines=lines,
            **kwargs
        )

        if nu.is_array_like(color):
            if type(color[0]) == str:
                #print(f"changing colors")
                color = np.array([mu.color_to_rgb(k) for k in color])
        scat.color = color

        if size is not None:
            #print(f"Setting size")
            scat.size = size

        
    
        widget_list = []

        if widgets_to_plot is None:
            widgets_to_plot = []

#         wid = ipvu.add_size_widget(scat,prefix="test_size")
#         widget_list.append(wid)
#         display(widgets.HBox(widget_list))
#         ipv.show()
#         return 
        idx_for_description = 0
        if plot_widgets:
            for j,w in enumerate(widgets_to_plot):
                try:
                    if j == idx_for_description:
                        description_prefix = widget_description_prefix
                    else:
                        description_prefix = None
                    curr_widget = getattr(
                        ipvu,
                        f"add_{w}_widget"
                    )(scat,description_prefix=description_prefix,**kwargs)
                    #print(f"Adding widget = {w}")
                except Exception as e:
                    idx_for_description += 1
                    #print(e)
                    pass
                else:
                    widget_list.append(curr_widget)
        
        if len(widget_list) > 0:
            display(widgets.HBox(widget_list))
    else:
        scat = None
        
    if len(array) <= 1:
        buffer = 5
    else:
        buffer = 0
    
    
    ipvu.set_axes_lim_from_fig(buffer,verbose = False)
    
    if alpha is not None:
        scat.material.transparent = True
        scat.color = mu.color_to_rgba(str(scat.color),alpha)

    if show_at_end:
        ipv.show()
        
    ipvu.set_axes_visibility(axis_visibility)
    
    if not individually_scale_each_axis:
        ipvu.set_axes_lim_to_cube()
        
    return scat
    
#def add_marker_selection()
def xyz_from_array(array):
    array = np.array(array).astype('float')
    return array[:,0],array[:,1],array[:,2]

#from python_tools from . import matplotlib_utils as mu
#from python_tools from . import mesh_utils as mhu
def plot_mesh(
    mesh,
    alpha = None,
    plot_widgets = True,
    widgets_to_plot = ("color",),
    show_at_end = True,
    new_figure = True,
    flip_y = True,
    **kwargs
    ):
    
    if nu.is_array_like(mesh):
        mesh = mhu.combine_meshes(mesh)
    return_mesh =  ipvu.plot_obj(
        mesh.vertices,
        #all possible inputs to functions
        triangles = mesh.faces,
        plot_widgets = True,
        widgets_to_plot = widgets_to_plot,
        show_at_end = show_at_end,
        new_figure = new_figure,
        flip_y = flip_y,
        alpha = alpha,
         **kwargs
    )
    
#     if alpha is not None:
#         return_mesh.material.transparent = True
#         return_mesh.color = mu.color_to_rgba(str(return_mesh.color),alpha)
#         print(f"return_mesh.color = {return_mesh.color}")
#         ipv.show()
    return return_mesh


def plot_mesh_with_scatter(
    mesh=None,
    scatter=None,
    mesh_alpha = 0.2,
    flip_y = True,
    ):
    
    
    new_figure = True
    if mesh is not None:
        mesh_obj = ipvu.plot_mesh(
                mesh,
                alpha=mesh_alpha,
                flip_y = flip_y,
                show_at_end=False,
                new_figure = True,
            )
        new_figure = False
    else:
        new_mesh = None
        
    sc_obj = ipvu.plot_scatter(
            scatter,
            flip_y = True,
            new_figure = new_figure,
            show_at_end=True,
        ) 
    
    return mesh_obj,sc_obj
    
def plot_scatter(
    array,
    plot_widgets = True,
    widgets_to_plot = ("size","marker","color"),
    show_at_end = True,
    new_figure = True,
    color = "red",
    size = 1,
    flip_y = True,
    axis_visibility=True,
    return_scatter = False,
    widget_description_prefix = None,
    **kwargs
    ):
    
    if type(array) == list:
        array = np.vstack(array)
        
    array=  array.reshape(-1,3)
    
    return_sc =  ipvu.plot_obj(
        array,
        plot_widgets = True,
        color = color,
        size = size,
        widgets_to_plot = widgets_to_plot,
        show_at_end = show_at_end,
        new_figure = new_figure,
        flip_y = flip_y,
        axis_visibility=axis_visibility,
        widget_description_prefix=widget_description_prefix,
         **kwargs
    )
    
    if return_scatter:
        return return_sc
    


#from python_tools from . import numpy_utils as nu

def plot_multi_scatters(
    scatters,
    color = "red",
    size = 1,
    plot_widgets = True,
    widgets_to_plot = ("size","marker","color"),
    flip_y = True,
    axis_visibility = False,
    verbose = False,
    show_at_end = True,
    new_figure = True,
    alpha = 1,
    ):
    
    """
    Purpose: To plot a group of synapses and have the 
    scatter controls to be able to control size for each color


    """
    
    if type(scatters) != list:
        scatters = [scatters]
    if not nu.is_array_like(color):
        color = [color]

    if len(color) != len(scatters):
        color = color*len(scatters)

    if not nu.is_array_like(size):
        size = [size]

    if len(size) != len(scatters):
        size = size*len(scatters)
        
    sum_points = np.sum([len(k) for k in scatters])
    if sum_points == 0:
        if verbose:
            print(f"No scatters to plot")
        if show_at_end:
            ipv.show()
        return

    all_scatter_obj = []
    for j,(sc,sc_c,sz) in enumerate(zip(
        scatters,
        color,
        size,
        )):
        
        show_at_end_inner = False
        new_figure_inner = False
        
        if (j == len(scatters) - 1) and show_at_end:
            show_at_end_inner = True
        if (j == 0) and new_figure:
            new_figure_inner = True
        sc_obj = ipvu.plot_scatter(
            sc,
            size = sz,
            color = sc_c,
            show_at_end = show_at_end_inner,
            new_figure = new_figure_inner,
            axis_visibility=axis_visibility,
            flip_y=flip_y,
            widgets_to_plot=widgets_to_plot,
            return_scatter=True,
            )
        
        all_scatter_obj.append(sc_obj)
        
    return all_scatter_obj


def print_ipv_cheatsheet():
    s = """
    Ipyvolume Cheat sheet: 

    1) Alt + Click and Hold: Can drag object
    2) Pinch: zoom in and Out
    3) Magnifying glass on + pinch: Zoom in and Out on place mouse hovering
    4) Double click and move mouse: rotate object
    5) gear icon: different resolutions
    6) Shift + Picture: Copies screenshot to clipboard
    """
    print(s)
    
def clear_selected(obj):
    obj.selected = None
    
def view_top_down(
    axis_visibility = True
):
    """
    Purpose: Make a top down view
    of the current figure
    
    
    azimuth (float) – rotation around the axis pointing up in degrees
    elevation (float) – rotation where +90 means ‘up’, -90 means ‘down’, in degrees
    distance (float) – radial distance from the center to the camera.


    """
    ipvu.set_axes_visibility(axis_visibility)
    ipv.pylab.view(azimuth=-90, elevation=90, distance=2)
    
def set_view(
    h_rotation = 0,
    v_rotation = 0, #between -90 (down) and 90 (up)
    distance = 1, #multiplier distance from axes
    center = None,
    radius = None,
    axis_visibility = None,
    ):
    
    """
    Purpose: To set the
    zoom level, and then 
    the proper rotation of an ipyvolume
    figure (to recreate )
    """
    
    ipvu.set_zoom(
        center_coordinate=center,
        radius = radius,
    )
    
    ipv.pylab.view(
        azimuth=h_rotation,
        elevation=v_rotation,
        distance=distance
    )
    
    if axis_visibility is not None:
        ipvu.set_axes_visibility(axis_visibility)
    
    
show_top_down = view_top_down
top_down = view_top_down
    
def plot_skeleton(nodes,edges,flip_y=True,color="green",**kwargs):
    ipvu.plot_obj(
        array = nodes,
        lines=edges,
        flip_y = flip_y,
        color = color,
        **kwargs
    )

    
def flip_y_func(array):
    array = np.array(array).astype("float")
    array[...,1] = -array[...,1]
    return array
def plot_quiver(
    centers,
    vectors,
    color_array = None,
    plot_colormap = True,
    size = 2,
    size_array = None,
    flip_y = True,
    new_figure = True,
    show_at_end = True,
    ):
    """
    Purpose: plotting quiver with a 
    color gradient determined by an 
    attribute (and a size gradient optional)
    """
    centers = np.array(centers).reshape(-1,3)
    vectors = np.array(vectors).reshape(-1,3)
    
    if flip_y:
        centers = flip_y_func(centers)
        vectors = flip_y_func(vectors)


    if new_figure:
        ipv.figure()
    quiver = ipv.quiver(
        *ipvu.xyz_from_array(centers,),
        *ipvu.xyz_from_array(vectors,)
    )
    
    if color_array is not None:
        quiver.color = mu.gradient_from_array(
            color_array,
            plot_colormap = True,
        )

    if size_array is not None:
        quiver.size = size_array
    else:
        slider = widgets.FloatSlider(min=0,max=10,value = size,description = "size")
        widgets.link((quiver,"size"),(slider,"value"))
        slider.value = size
        display(slider)

    if show_at_end:
        ipv.show()
        
def plot_line_segments(
    array,
    lines=None,
    plot_widgets = True,
    widgets_to_plot = ("size","marker","color"),
    show_at_end = True,
    new_figure = True,
    color = "red",
    size = 1,
    flip_y = True,
    axis_visibility=True,
    return_obj = False,
    widget_description_prefix = None,
    **kwargs
    ):
    
    if type(array) == list:
        array = np.vstack(array)
    
    return_sc =  ipvu.plot_obj(
        array,
        lines=lines,
        plot_type = "line_segments",
        plot_widgets = plot_widgets,
        color = color,
        size = size,
        widgets_to_plot = widgets_to_plot,
        show_at_end = show_at_end,
        new_figure = new_figure,
        flip_y = flip_y,
        axis_visibility=axis_visibility,
        widget_description_prefix=widget_description_prefix,
         **kwargs
    )
    
    if return_obj:
        return return_sc
    
plot_quiver_with_gradients = plot_quiver

# --------------- helper functions with making movies ----
def rotation_movie_export():
    # create 2d grids: x, y, and r
    u = np.linspace(-10, 10, 25)
    x, y = np.meshgrid(u, u)
    r = np.sqrt(x**2+y**2)
    print("x,y and z are of shape", x.shape)
    # and turn them into 1d
    x = x.flatten()
    y = y.flatten()
    r = r.flatten()
    print("and flattened of shape", x.shape)


    # create a sequence of 15 time elements
    time = np.linspace(0, np.pi*2, 15)
    z = np.array([(np.cos(r + t) * np.exp(-r/5)) for t in time])
    print("z is of shape", z.shape)

    # draw the scatter plot, and add controls with animate_glyphs
    ipv.figure()
    s = ipv.scatter(x, z, y, marker="sphere")
    ipv.animation_control(s, interval=200)
    ipv.ylim(-3,3)
    ipv.show()

    # Now also include, color, which containts rgb values
    color = np.array([[np.cos(r + t), 1-np.abs(z[i]), 0.1+z[i]*0] for i, t in enumerate(time)])
    size = (z+1)
    print("color is of shape", color.shape)


    #This is commented out, otherwise it would run on readthedocs
    def set_view(figure, framenr, fraction):
        ipv.view(fraction*360)
        #s.size = size * (2+0.5*np.sin(fraction * 6 * np.pi))
    ipv.movie('wave.mp4', set_view, fps=20, frames=40)
    
#import ipywidgets
 
#import os
def movie(
    func,
    filename="movie.mp4", 
    #function=_change_azimuth_angle, 
    fps=30, 
    frames=30,
    endpoint=False, 
    cmd_template_ffmpeg="ffmpeg -y -r {fps} -i {tempdir}/frame-%5d.png -vcodec h264 -pix_fmt yuv420p {filename}",
    cmd_template_gif="convert -delay {delay} {loop} {tempdir}/frame-*.png {filename}",
    gif_loop=0,
    width = 1920,
    height = 1080,
    **kwargs):
    """Create a movie (mp4/gif) out of many frames

    If the filename ends in `.gif`, `convert` is used to convert all frames to an animated gif using the `cmd_template_gif`
    template. Otherwise `ffmpeg is assumed to know the file format`.

    Example:

    >>> def set_angles(fig, i, fraction):
    >>>     fig.angley = fraction*np.pi*2
    >>> # 4 second movie, that rotates around the y axis
    >>> p3.movie('test2.gif', set_angles, fps=20, frames=20*4,
            endpoint=False)

    Note that in the example above we use `endpoint=False` to avoid to first and last frame to be the same

    :param str f: filename out output movie (e.g. 'movie.mp4' or 'movie.gif')
    :param function: function called before each frame with arguments (figure, framenr, fraction)
    :param fps: frames per seconds
    :param int frames: total number of frames
    :param bool endpoint: if fraction goes from [0, 1] (inclusive) or [0, 1) (endpoint=False is useful for loops/rotatations)
    :param str cmd_template_ffmpeg: template command when running ffmpeg (non-gif ending filenames)
    :param str cmd_template_gif: template command when running imagemagick's convert (if filename ends in .gif)
    :param gif_loop: None for no loop, otherwise the framenumber to go to after the last frame
    :return: the temp dir where the frames are stored
    """
    movie_filename = filename
    import tempfile
    tempdir = tempfile.mkdtemp()
    output = ipywidgets.Output()
    display(output)
    fig = ipv.gcf()
    for i in range(frames):
        with output:
            fraction = i / (frames - 1. if endpoint else frames)
            func(fig, i, fraction)
            frame_filename = os.path.join(tempdir, "frame-%05d.png" % i)
            ipv.savefig(frame_filename, output_widget=output,width=width,height=height,**kwargs)
    with output:
        if movie_filename.endswith(".gif"):
            if gif_loop is None:
                loop = ""
            else:
                loop = "-loop %d" % gif_loop
            delay = 100 / fps
            cmd = cmd_template_gif.format(delay=delay, loop=loop, tempdir=tempdir, filename=movie_filename)
        else:
            cmd = cmd_template_ffmpeg.format(fps=fps, tempdir=tempdir, filename=movie_filename)
        print(cmd)
        os.system(cmd)
    return tempdir

def save_fig(
    filename = None,
    scale = 1,
    default_dim = 1024,
    width = None,
    height = None):
    
    if width is None:
        width = default_dim*scale
    if height is None:
        height = default_dim*scale
    save_config_ex = dict(
        width = width,
        height = height
    )
    
    if filename is None:
        filename = f"./{np.random.randint(10000)}_ipv_pic.png"
    if filename[-4] != ".":
        filename = f"{filename}.png"
    ipv.savefig(
        filename,
        **save_config_ex,
    )
    
def set_axes_lim_equal_width_from_array(
    array,
    flip_y = True,
    verbose = False,
    buffer = 0,
    ):
    
    axes_lim = nu.axes_lim_equal_width(
        array = array,
        flip_y = flip_y,
        verbose = verbose,
        buffer=buffer,
    ).T

    set_axes_lim(
        xlim=axes_lim[0],
        ylim=axes_lim[1],
        zlim=axes_lim[2],
    )
    
    
"""
Example of how to use ipvu.movie to make a movie about a neuron: 

exc_name = "864691135494192528_0"
conu.visualize_graph_connections_by_method(
    G,
    segment_ids = [bpc_name,bc_name,exc_name],
    segment_ids_colors = ["skyblue","orange","black"],
    method = "meshafterparty",
    plot_gnn=False,
    synapse_color = "red",
    plot_soma_centers=False,
    
    plot_synapse_skeletal_paths = True,
    plot_proofread_skeleton = False,
    
    synapse_path_presyn_color='plum',
    synapse_path_postsyn_color='lime',
    
    transparency = 0.8,
    
    synapse_scatter_size=1.4,
    synapse_path_scatter_size=0.7,
    
)


# apt install ffmpeg
def set_view(figure, framenr, fraction):
    ipv.view(fraction*360,distance = 1)
    #s.size = size * (2+0.5*np.sin(fraction * 6 * np.pi))
    
fps = 60
n_sec = 10
ipvu.movie(
    filename = f'./bpc_mc_23p_rotation.mp4', 
    func=set_view,
    fps=fps, 
    frames=fps*n_sec,
    cmd_template_ffmpeg='ffmpeg -y -r {fps} -i {tempdir}/frame-%5d.png -vcodec h264 -pix_fmt yuv420p {filename}',
)

"""

def lighting_parameters():
    print("""
    ambient_coefficient – lighting parameter
    diffuse_coefficient – lighting parameter
    specular_coefficient – lighting parameter
    specular_exponent – lighting parameter
    """)

#from python_tools from . import ipyvolume_utils as ipvu
    
    
    
    

#--- from python_tools ---
from . import matplotlib_utils as mu
from . import mesh_utils as mhu
from . import numpy_utils as nu

from . import ipyvolume_utils as ipvu