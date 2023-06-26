
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#import matplotlib.gridspec as gridspec
#import seaborn as sns
#import numpy as np

class SeabornFig2Grid():
    """
    Purpose: Allows some seabonrn plots to be subplots
    
    source: https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
    """
    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        try:
            self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        except:
            pass
        try:
            self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])
        except:
            pass

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())
        

        
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import seaborn as sns;# sns.set()
#from python_tools.seaborn_utils import SeabornFig2Grid

def example_SeabornFig2Grid():
    iris = sns.load_dataset("iris")
    tips = sns.load_dataset("tips")

    # An lmplot
    g0 = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, 
                    palette=dict(Yes="g", No="m"))
    # A PairGrid
    g1 = sns.PairGrid(iris, hue="species")
    g1.map(plt.scatter, s=5)
    # A FacetGrid
    g2 = sns.FacetGrid(tips, col="time",  hue="smoker")
    g2.map(plt.scatter, "total_bill", "tip", edgecolor="w")
    # A JointGrid
    g3 = sns.jointplot("sepal_width", "petal_length", data=iris,
                       kind="kde", space=0, color="g")


    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(2, 2)

    mg0 = SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid(g2, fig, gs[3])
    mg3 = SeabornFig2Grid(g3, fig, gs[2])

    gs.tight_layout(fig)
    #gs.update(top=0.7)

    plt.show()
    
def example_gridspec_from_existing_ax():
    """
    Pseudocode: 
    1) Create the figure
    2) Great the gripspec
    3) Use the seabornfig2grid to add the ax to figure 
    and reference the certain gripspec
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=n_rows,ncols=col_width)
    for ax_i,ax_obs in enumerate(axes_objs):
        row_idx = n_rows_per_ax*(ax_i)+1
        start_idx = col_width*(row_idx-1) + 1
        for i,ax in enumerate(ax_obs):
            mu.set_n_ticks_from_ax(ax,x_nticks=x_nticks,y_nticks=y_nticks)
            row_i,idx = row_idx + int(np.floor(i/col_width)),start_idx + i
            #print(f"{(n_rows,col_width,start_idx + i)}")
            snsu.SeabornFig2Grid(ax, fig, gs[start_idx + i-1])

            if plot_figure_letters:
                ax.ax_joint.text(letter_x, letter_y, string.ascii_lowercase[start_idx + i-1], transform=ax.ax_joint.transAxes, 
                    size=letters_fontsize, weight='bold')

    gs.tight_layout(fig)
    plt.show()

#from python_tools.seaborn_utils import SeabornFig

