
from . import numpy_dep as np

#from . import numpy_dep as np
def start_end_frame_from_start_end_frac(
    start,
    end,
    n_frames = None,
    fps=None,
    n_sec=None,
    verbose = False,
    return_dict = True,
    ):
    """
    Purpose: To calculate the start and 
    end frame for a fade based on the fraction
    of time you want it starting and ending
    """

    if n_frames is None:
        n_frames = fps*n_sec
        
    fade_in_start_frame = np.floor(n_frames*start).astype('int')
    fade_in_end_frame = np.floor(n_frames*end).astype('int')

    n_fade_frames = fade_in_end_frame - fade_in_start_frame

    if verbose:
        print(f"fade_in_start_frame= {fade_in_start_frame}, fade_in_end_frame = {fade_in_end_frame}")
        print(f"n_fade_frames = {n_fade_frames}")

    if return_dict:
        return dict(fade_in_start_frame=fade_in_start_frame,
                   fade_in_end_frame=fade_in_end_frame,
                   n_fade_frames=n_fade_frames) 
    else:
        return fade_in_start_frame,fade_in_end_frame



def example_rotating_motif():
    from neurd import connectome_utils as conu
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
    
    
#from datasci_tools import ipyvolume_movie_utils as ipvm



#--- from datasci_tools ---
from . import ipyvolume_utils as ipvu

from . import ipyvolume_movie_utils as ipvm