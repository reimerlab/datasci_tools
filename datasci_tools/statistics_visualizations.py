'''



Provides a great example of how to visualize
projections from 1d to 3d: 

http://at-compute004.ad.bcm.edu:8888/notebooks/neuron_mesh_tools/Auto_Proofreading/Soma_Filtering/Soma_Soma_Merger_Detector.ipynb#






'''
from sklearn.metrics import roc_curve
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from . import numpy_dep as np


#from datasci_tools import pretty_print_confusion_matrix as p_cm
def plot_pretty_confusion_matrix(y_true,
                                 y_pred,
                                 labels=None,
    figsize = (20,20),
    cell_fontsize = 20,
    axes_fontsize=30,
    title='Confusion matrix',
    title_fontsize=30,
    ticklabel_fontsize=15,
    cmap="Oranges",
    **kwargs,
    ):
    """
    Purpose: To take 2 arrays 
    
    """
    if labels is None:
        labels = list(np.union1d(np.unique(y_true),np.unique(y_pred)))
    p_cm.plot_confusion_matrix_from_data(y_true,
                                         y_pred,
                                         figsize=figsize,
                                    columns=labels,
                                    fz=cell_fontsize,
                                axes_fontsize=axes_fontsize,
                                title=title,
                                title_fontsize=title_fontsize,
                                ticklabel_fontsize=ticklabel_fontsize,
                                cmap=cmap,
                             **kwargs)
    
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib
#from . import numpy_dep as np

def plot_heatmap(#need to order by 
    df=None,
    input_row_names=None,
    input_col_names=None,
    data=None,
    fontsize_cell=None,
    
    y_label='Statistics',
    x_label = "Type",
    title = "Autoproofreading Validation"):


    """
    Purpose: To plot the scores
    of the presyn, postsyn and both
    
    Needs to be of the form:
    Type | Stat1 | Stat2 | Stat 3
    -----------------------------
    Type 1
    Type 2
    Type 3

    """
    if input_row_names is None:
        input_row_names = list(df.index)
    if input_col_names is None:
        input_col_names = list(df.columns)
    synapse_types= [k.replace("_"," ") for k in input_row_names]
    score_types = input_col_names
    if df is not None:
        score_df = df

        collapsed_df = score_df.loc[synapse_types]
        collapsed_df = collapsed_df.loc[:,score_types]
        median_accuracies = collapsed_df.to_numpy()
    else:
        median_accuracies = data
    #setting the text size
    
    
    accuracies = median_accuracies.T
    cm = accuracies
    normalize = True
    cmap=plt.cm.Blues
    #now graph the results for all the categories
    

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(accuracies, interpolation='nearest', cmap=cmap,vmin=0.5,vmax=1)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(accuracies.shape[1]),
           yticks=np.arange(accuracies.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=synapse_types, yticklabels=score_types,
           title=title,
           ylabel=y_label,
           xlabel=x_label)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = np.nanmax(cm) / 2.
    thresh = 0.6
    #print("threshold = " + str(thresh))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            #print("cm[i,j] = " + str(cm[i,j]))
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                   fontsize=fontsize_cell)
    fig.tight_layout()
    plt.rcParams.update({'font.size': 2})
    fig.set_size_inches(7, 4)
    ax.grid(False)
    
    
    
#from datasci_tools import pandas_utils as pu
#import matplotlib.pyplot as plt

def n_rows_from_n_plots(n_plots,
                       axes_per_row = 4):
    if type(n_plots) != int:
        n_plots = len(n_plots)
    return np.ceil(n_plots/axes_per_row).astype("int")

def row_column_from_index(index,axes_per_row):
    row_idx = np.floor(index/axes_per_row).astype("int")
    return row_idx,index - row_idx*axes_per_row

def plot_table_histograms_from_divided_tables(tables_to_plot,
                          tables_labels,
                          fig_title=None,
                         axes_per_row=4,
                         alpha=0.4,
                         n_bins=50,
                        figure_width=18.5,
                        figure_height=None,
                        figure_height_per_row=2.5,
                        ):
    """
    To plot the distributions of different features between different tables
    """

    n_rows= sviz.n_rows_from_n_plots(len(tables_to_plot[0].columns),axes_per_row)
    curr_table = tables_to_plot[0]
    
    if figure_height is None:
        figure_height = figure_height_per_row*n_rows

    
    fig,axes = plt.subplots(n_rows,axes_per_row)
    #fig.set_size_inches(18.5, 10.5)
    
    #print(f"figure_width,figure_height = {figure_width,figure_height}")
    fig.set_size_inches(figure_width,figure_height)
    
    
    
    
    if not fig_title is None:
        fig.title(fig_title)

    
    for j,col_title in enumerate(curr_table):
        #row = np.floor(j/axes_per_row).astype("int")
        row,column = sviz.row_column_from_index(j,axes_per_row)
        try:
            ax = axes[row,column]
        except:
            ax = axes[column]
        ax.set_title(col_title)
        
        for curr_table,curr_table_name in zip(tables_to_plot,tables_labels):
            curr_data = curr_table[col_title].to_numpy()
            ax.hist(curr_data,bins=n_bins,label=curr_table_name,alpha=alpha,density=True)
            
        ax.legend()
        
    fig.tight_layout()

#from datasci_tools import pandas_utils as pu

def plot_table_statistics(df,
                          df_plotting_func,
                          category_column="label",
                          categories_to_plot = None,
                         fig_title=None,
                         axes_per_row=4,
                         alpha=0.4,
                          columns_to_ignore = None,
                          columns_to_plot = None,
                          **kwargs
                          
                         ):
    if columns_to_ignore is not None:
        df = pu.delete_columns(df,columns_to_ignore)
        
    if columns_to_plot is not None:
        df = df[[category_column] + columns_to_plot]
    
    unique_categories = np.unique(df[category_column].to_numpy())
    
    df_list = []
    for k in unique_categories:
        if categories_to_plot is not None:
            if k not in categories_to_plot:
                unique_categories = unique_categories[unique_categories!= k]
                continue
        new_df = df[df[category_column] == k]
        new_df = pu.delete_columns(new_df,[category_column])
        df_list.append(new_df)
    
    #df_list = [df[df[category_column] == k] for k in unique_categories]
    df_plotting_func(tables_to_plot=df_list,
                                             tables_labels=unique_categories,
                                             fig_title=fig_title,
                                             axes_per_row=axes_per_row,
                                             alpha=alpha,
                                             **kwargs)

def plot_table_histograms(df,
                          category_column="label",
                          categories_to_plot = None,
                         fig_title=None,
                         axes_per_row=4,
                         alpha=0.4,
                          columns_to_ignore = None,
                         n_bins=50,
                          columns_to_plot = None,
                         ):
    """
    Purpose: To take a table that has labeled  data
    in a column and then to plot the distributions 
    of all of the features
    
    Pseudocode:
    1) To divide the current table up into multiple tables
    based on the number of categories in the category column
    2) Send those tables to the divided table function for plotting 

    """
    plot_table_statistics(df,
                          df_plotting_func=sviz.plot_table_histograms_from_divided_tables,
                          category_column=category_column,
                          categories_to_plot = categories_to_plot,
                         fig_title=fig_title,
                         axes_per_row=axes_per_row,
                         alpha=alpha,
                          columns_to_ignore = columns_to_ignore,
                          columns_to_plot = columns_to_plot,
                          n_bins=n_bins
                          
                         )
    
#from datasci_tools import numpy_utils as nu
#import matplotlib.pyplot as plt

def plot_table_scatter_statistics_from_divided_tables(
    tables_to_plot,
    tables_labels,
    figure_height = None,
    fig_title = None,
    stats_combinations = None,
    axes_per_row = 4,
    figure_height_per_row=2.5,
    figure_width=18.5,
    verbose = False,
    alpha=0.4,
    n_plot_dim = 2,
    ):
    """
    Purpose: To plot a combination of two statistics from table

    Pseudocode:
    1) If no combination of statistics are provided then generate all combinations (this will help determine number of rows)
    2) Compute the number of rows will need for all of the stats
    3) For each combination of statistics:
    a) Get the data



    """
    
    if stats_combinations is None:
        stats_combinations = nu.all_unique_choose_k_combinations(tables_to_plot[0].columns,n_plot_dim)
    else:
        n_plot_dim = len(stats_combinations[0])

    if n_plot_dim == 3:
        projection = "3d"
    else:
        projection = None

    n_rows = sviz.n_rows_from_n_plots(stats_combinations,
                           axes_per_row = axes_per_row)

    if figure_height is None:
        figure_height = figure_height_per_row*n_rows

    if verbose:
        print(f"stats_combinations = {stats_combinations}")
        print(f"n_rows = {n_rows}")
        print(f"figure_height= {figure_height}")

    #fig,axes = plt.subplots(n_rows,axes_per_row,)#projection='2d')
    fig = plt.figure()
    
    #fig.set_size_inches(18.5, 10.5)

    #print(f"figure_width,figure_height = {figure_width,figure_height}")
    fig.set_size_inches(figure_width,figure_height)


    

    if not fig_title is None:
        fig.title(fig_title)

    for j,data_names in enumerate(stats_combinations):
        #row,column = sviz.row_column_from_index(j,axes_per_row)
        ax = fig.add_subplot(n_rows, axes_per_row, j+1, projection=projection)
#         try:
#             ax = axes[row,column]
#         except:
#             ax = axes[column]
        col_title = " vs ".join(np.flip(data_names))
        ax.set_title(col_title)

        for curr_table,curr_table_name in zip(tables_to_plot,tables_labels):
            data = [curr_table[k].to_numpy() for k in data_names]
            ax.scatter(*data,label = curr_table_name,alpha = alpha)
            ax.set_xlabel(data_names[0])
            ax.set_ylabel(data_names[1])
            
            if n_plot_dim > 2:
                ax.set_zlabel(data_names[2])

        ax.legend()
        
    fig.tight_layout()
        
def plot_table_scatters(df,
                        stats_combinations=None,
                          category_column="label",
                          categories_to_plot = None,
                         fig_title=None,
                         axes_per_row=4,
                         alpha=0.4,
                          columns_to_ignore = None,
                        **kwargs
                         ):
    """
    Purpose: To take a table that has labeled  data
    in a column and then to plot the distributions 
    of all of the features
    
    Pseudocode:
    1) To divide the current table up into multiple tables
    based on the number of categories in the category column
    2) Send those tables to the divided table function for plotting 

    """
    
    plot_table_statistics(df,
                          df_plotting_func=sviz.plot_table_scatter_statistics_from_divided_tables,
                          category_column=category_column,
                          categories_to_plot = categories_to_plot,
                         fig_title=fig_title,
                         axes_per_row=axes_per_row,
                         alpha=alpha,
                          columns_to_ignore = columns_to_ignore,
                          stats_combinations=stats_combinations,
                          **kwargs
                          
                         )
    

def scatter_3D_ipv(scatters,
                  scatters_colors = [],
                  scatter_size = 10,
                  buffer = 1,
                 flip_y=False,
                 axis_box_off = False,):

    """
    Purpose: To plot a 3D scatter plot
    
    sviz.scatter_3D_ipv(scatters=[np.array([1,1,1]),np.array([2,2,2])],
                   scatter_size=10)
    
    """
    from neurd import neuron_visualizations as nviz
    nviz.plot_objects(
        scatters=scatters,
            scatters_colors=scatters_colors,
            scatter_size=scatter_size,
            buffer = buffer,
            flip_y = flip_y,
            axis_box_off=axis_box_off,
            adaptive_min_max_limits = False
    )
    
#from datasci_tools import matplotlib_utils as mu

def heatmap_3D(
    values,
    coordinates,
    n_bins = 20,
    feature_name = "feature_name",
    scatter_size = 0.5,
    plot_highest_bin_value = False,
    axis_box_off = False,
    bin_type = "equal_width",
    **kwargs
    ):
    """
    Purpose: To plot a 3D heatmap of a feature with values and coordinates
    """
    from neurd import neuron_visualizations as nviz
    
    curr_data= values
    curr_centers = coordinates

    data_dict,color_dict,bin_spacing_to_return = mu.divided_data_into_color_gradient(
    data = curr_data,
    n_bins = n_bins,
    bin_type=bin_type,
    verbose = False
    )

    scatters = []
    scatters_colors = []

    for bin_idx,data_idx in data_dict.items():
        scatters.append(curr_centers[data_idx].reshape(-1,3))
        scatters_colors.append(color_dict[bin_idx])

    x_values = bin_spacing_to_return
    x_colors = np.array(list(color_dict.values())).reshape(-1,3)
    

    if not plot_highest_bin_value:
        x_values = x_values[:-1]
        x_colors = x_colors[:-1]
        
    y = np.ones(x_values.shape)
    
    fig,ax = plt.subplots(figsize=(5,2))
    plt.scatter(x_values,y,c = x_colors,s = 200)
    plt.xlabel(feature_name)
    print(f"Color Scale")
    plt.show()

    nviz.plot_objects(
    scatters=scatters,
    scatters_colors=scatters_colors,
    scatter_size=scatter_size,
    axis_box_off = axis_box_off,
    **kwargs
    
    )
    
def heatmap_2D(
    values,
    coordinates,
    n_bins = 20,
    feature_name = "feature_name",
    bin_type = "equal_width",
    plot_highest_bin_value = False,
    figsize=(10,10),
    alpha = 0.3,
    size = 20,
    ):
    
    data_dict,color_dict,bin_spacing_to_return = mu.divided_data_into_color_gradient(
    data = values,
    n_bins = n_bins,
    bin_type=bin_type,
    verbose = False
    )
    
    x_values = bin_spacing_to_return
    x_colors = np.array(list(color_dict.values())).reshape(-1,3)
    
    if not plot_highest_bin_value:
        x_values = x_values[:-1]
        x_colors = x_colors[:-1]
    
    y = np.ones(x_values.shape)
    
    fig,ax = plt.subplots(figsize=(5,2))
    plt.scatter(x_values,y,c = x_colors,s = 200)
    plt.xlabel(feature_name)
    print(f"Color Scale")
    plt.show()
    
    fig,ax = plt.subplots(1,1,figsize=figsize)
    for bin_idx,data_idx in data_dict.items():
        ax.scatter(coordinates[data_idx][:,0],
                   coordinates[data_idx][:,1],
                    c = color_dict[bin_idx],
                      alpha = alpha,
                        s = size,)
        
    plt.show()
    
    
#import matplotlib.pyplot as plt
#from sklearn.metrics import roc_curve, auc

def plot_roc(
    fpr,
    tpr,
    auc_score=None,
    color="darkorange",
    line_width = 2,):
    
    if auc_score is None:
        auc_score = auc(fpr,tpr)
        
    plt.plot(
    fpr,
    tpr,
    color=color,
    lw=line_width,
    label="ROC curve (area = %0.2f)" % auc_score,
    )
    
    plt.plot([0, 1], [0, 1], color="navy", lw=line_width, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.show()


    
#from datasci_tools import statistics_visualizations as sviz



#--- from datasci_tools ---
from . import matplotlib_utils as mu
from . import numpy_utils as nu
from . import pandas_utils as pu
from . import pretty_print_confusion_matrix as p_cm

from . import statistics_visualizations as sviz