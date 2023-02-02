"""
Purpose: To make pandas table manipulation easier
"""

"""
Some useful functions: 
#sets the index to a specific column and doesn't return dataframe but modifies the existing one
df.set_index('Mountain', inplace=True) 

Selecting Data: 
i) selecting columns
- all columns are attribtues so can select with the "." (Ex: df.myCol)
- if there are wierd spaces could do getattr(df, 'Height (m)')
- BEST WAY: df['Height (m)'] because can select multiple columns at a time:
        df[['Height (m)', 'Range', 'Coordinates']]

ii) Selecting Rows (with slicing): 
- Can do that with index: a[start:stop:step]  (Ex: df[2:8:2])
- if you changed he index to be a certain column with non-numeric values: df['Lhotse':'Manaslu']
        slicing is inclusive at both start and end

iii) Specific data points: 
a) ** df.iloc** When just want to use numbers to reference
- df.iloc[rows, columns] for rows/col it can be: single int, list of ints, slices, : (for all)
  Ex: df.iloc[:, 2:6]** when 
  
b) ** df.loc where can use the names of things and not just numbers**
- df.loc[rows,columns]
   rows: index label, list of index labels, slice of index labels,:
   cols; singl column name, list of col names, slice of col names, :
   --> remember if do slice then it is inclusive on both start and stop
   
Ex: df.loc[:,'Height (m)':'First ascent']


c) Boolean selection: 
- can select rows with true/false mask--> 1) Create true false mask, put inside df[ ]
Ex: df[df['Height (m)'] > 8000]

For putting multiple conditions together the operators are: &,|,~

Have to use parenthesis to seperate multiple conditions: 
df[(df['Height (m)'] > 8000) & (df['Range']=='Mahalangur Himalaya')]

- Can use the loc operator to apply the mask and then select subset of columns: 
df.loc[(df['Height (m)'] > 8000) & (df['Range']=='Mahalangur Himalaya'), 'Height (m)':'Range']

Can also select the columns using a boolean operator: 
- col_criteria = [True, False, False, False, True, True, False]
  df.loc[df['Height (m)'] > 8000, col_criteria]
  
  
- To delete a column:
del neuron_df["random_value"]




------------------------------------pd.eval, df.eval df.query ------------------------------
Purpose: All of these functions are used to either
1) create masks of your rows
2) filter for specific rows 
3) make new data columns

pd.query:
- Can reference other dataframes by name and their columns with the attribute "."
pd.eval("df1.A + df2.A")   # Valid, returns a pd.Series object
pd.eval("abs(df1) ** .5")  # Valid, returns a pd.DataFrame object

- can evaluate conditional expressions: 
pd.eval("df1 > df2")        
pd.eval("df1 > 5")    
pd.eval("df1 < df2 and df3 < df4")      
pd.eval("df1 in [1, 2, 3]")
pd.eval("1 < 2 < 3")


Arguments that are specific:
- can use & or and (same thing)

** parser = "python" or "pandas" **
pandas: evaluates some Order of Operations differently
> is higher proirity than &
== is same thing as in  Ex: pd.eval("df1 in [1, 2, 3]") same as pd.eval("df1 == [1, 2, 3]")

python: if want traditional rules


** engine = "numexpr" or "python" ***

python: 
1) can do more inside your expressions but it is not as fast
Example: 
df = pd.DataFrame({'A': ['abc', 'def', 'abacus']})
pd.eval('df.A.str.contains("ab")', engine='python')

2) can reference variables with dictionary like syntax (not worth it)



---------- things can do with pd.eval -----------
1) Pass in dictionary to define variables that not defined 
        (otherwise uses global variable with same name)

pd.eval("df1 > thresh", local_dict={'thresh': 10})


---------- things can do with pd.eval -----------
1) only have to write column names in queries because only applied
to one dataframe so don't need to put name in front

df1.eval("A + B")

2) Have to put @ in front of variables to avoid confusion with column names
A = 5
df1.eval("A > @A") 


3) Can do multiline queries and assignments:

df1.eval('''
E = A + B
F = @df2.A + @df2.B
G = E >= F
''')

Can still do the local dict:
returned_df.query("n_faces_branch > @x",local_dict=dict(x=10000))

----------- Difference between df1.eval and df1.query -----------
if you are returning a True/False test, then df1.eval will just
return the True/False array whereas df1.query will go one step further
and restrict the dataframe to only those rows that are true

AKA: df1.eval is an intermediate step of df1.query 
(could use the df1.eval output to restrict rows by df1[output_eval])


---------------------------Examples--------------------------

Ex_1: How to check if inside a list
list_of_faces = [1038,5763,7063,11405]
returned_df.query("n_faces_branch in @list_of_faces",local_dict=dict(list_of_faces=list_of_faces))

Ex_2: Adding Ands/Ors

list_of_faces = [1038,5763,7063,11405]
branch_threshold = 31000
returned_df.query("n_faces_branch in @list_of_faces or skeleton_distance_branch > @branch_threshold",
                  local_dict=dict(list_of_faces=list_of_faces,branch_threshold=branch_threshold))
                  
Ex 2: of how to restrict a column to being a member or not being in a list
cell_df.query("not segment_id in @error_segments",local_dict=dict(error_seegments=error_segments))


--- 5/4 -----
How to query for null values: 
df.query('value < 10 | value.isnull()', engine='python')

ORRRRRR you could just check that the col not equal to itself (will get none values)
filt_df.query("cell_type_coarse != cell_type_coarse")
 

--- 6/14: Review of how to check if in list -------

df = pd.DataFrame({"a":[1,2,3,4],"b":[2,3,4,5]})
curr_list = [1,2]
df.query("a in @my_list",
        local_dict = dict(my_list=curr_list))
        
        
-- 12/8 : how to modify something in place using index and column

idx_to_change = df.query("axon == 0").index
df.loc[idx_to_change,"synapse_density"] = df.loc[idx_to_change,"synapse_density"]*df.loc[idx_to_change,"n_synapses_post"] / df.loc[idx_to_change,"n_synapses"]
df.loc[idx_to_change,"n_synapses"] =  df.loc[idx_to_change,"n_synapses_post"]
df.loc[idx_to_change,"n_synapses_pre"]  =  0
df.loc[idx_to_change,"synapse_density_pre"]  =  0


----- PIVOT TABLES ----

Pivot table:
Index: The rows you want for your pivot table
values: columns you want (wha tthe aggre will work on )
aggfun: can be a list of things on how you want that combination to be combined
columns: if you wanted to break the data further by columns and then the value
in the boxes is just what you specify


"""
pd_source = "pandas"

from pandas import util

def random_dataframe():
    return util.testing.makeDataFrame()

def random_dataframe_with_missing_data():
    util.testing.makeMissingDataframe()
    
def dataframe_to_row_dicts(df,):
    if pu.is_series(df):
        return df.to_dict()
    else:
        return df.to_dict(orient='records')

df_to_dicts = dataframe_to_row_dicts

def df_to_dicts_index_key(
    df,
    index_column = None,
    ):
    if index_column is not None:
        df = df.set_index(index_column)
    return df.to_dict("index")

def nans_in_column(df):
    return pd.isnull(df).any()

def n_nans_per_column(df):
    return df.isnull().sum()

def n_nans_total(df):
    return df.isnull().sum().sum()

def surpress_scientific_notation():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    
import pandas as pd
import pandasql

def set_pd_from_modin():
    """
    This library supposedly speeds up pandas functions
    by using multiple cores at once (made some things longer tho...)
    """
    global pd
    import modin.pandas as pd_modin
    pd = pd_modin
    
def set_pd_from_pandas():
    global pd
    import pandas as pdu
    pd = pdu

def restrict_pandas(df,index_restriction=[],column_restriction=[],value_restriction=""):
    """
    Pseudocode:
    How to specify:
    1) restrict rows by value (being equal,less than or greater) --> just parse the string, SO CAN INCLUDE AND
    2) Columns you want to keep
    3) Indexes you want to keep:
    
    Example: 
    
    index_restriction = ["n=50","p=0.25","erdos_renyi_random_location_n=100_p=0.10"]
    column_restriction = ["size_maximum_clique","n_triangles"]
    value_restriction = "transitivity > 0.3 AND size_maximum_clique < 9 "
    returned_df = restrict_pandas(df,index_restriction,column_restriction,value_restriction)
    returned_df
    
    Example of another restriction = "(transitivity > 0.1 AND n_maximal_cliques > 10) OR (min_weighted_vertex_cover_len = 18 AND size_maximum_clique = 2)"
    
    
    #how to restrict by certain value in a column being a list 
    df[~df['stn'].isin(remove_list)]
    """
    new_df = df.copy()
    
    
    if len(index_restriction) > 0:
        list_of_indexes = list(new_df[graph_name])
        restricted_rows = [k for k in list_of_indexes if len([j for j in index_restriction if j in k]) > 0]
        #print("restricted_rows = " + str(restricted_rows))
        new_df = new_df.loc[new_df[graph_name].isin(restricted_rows)]
        
    #do the sql string from function:
    if len(value_restriction)>0:
        s = ("SELECT * "
            "FROM new_df WHERE "
            + value_restriction + ";")
        
        #print("s = " + str(s))
        new_df = pandasql.sqldf(s, locals())
        
    #print(new_df)
    
    #restrict by the columns:
    if len(column_restriction) > 0:
        #column_restriction.insert(0,graph_name)
        new_df = new_df[column_restriction]
    
    return new_df

import re
def rewrite_sql_functions_in_query(query,
                                  function_names = None,
                                   verbose = False,
                                   **kwargs
                                  ):
    """
    Purpose: To replace a sql keyword function
    inside a query so it will work with sql querying
    """
    def convert_func(match_obj):
        return f"(SELECT {match_obj.group(1).upper()}({match_obj.group(2)}) FROM df)"
    
    if function_names is None:
        function_names = ["MAX","MIN"]
        
    for func_name in function_names:
        if verbose:
            print(f"\n---Working on {func_name}---")
        
        if verbose:
            print(f"before replace:\n {query}")
            
        sql_pattern = re.compile(fr"({func_name}|{func_name.lower()}|{func_name.upper()})\(([A-Za-z0-9_\s]+)\)")
        query = sql_pattern.sub(convert_func,query)
        
        if verbose:
            print(f"after replace:\n {query}")
            
    return query
        
    
    
        
import math
def query(df,query,**kwargs):
    """
    PUrpose: To query a dataframe using an query 
    that would work with sql
    
    Ex:
    import pandas_utils as pu
    import pandas as pd

    curr_dicts = [dict(x = 5,y=10),dict(x = 7,y=11)]
    df = pd.DataFrame.from_records(curr_dicts)
    s = "(x == MAX( x )) or (y = min(y))"

    #s = pu.rewrite_sql_functions_in_query(s)
    pu.query(df,s)
    """
    df_orig = df.copy()
    index_name = "xyzilij"
    try:
        

        s = ("SELECT * "
            "FROM df WHERE "
            + query + ";")
        s_cp = s
        s = pu.rewrite_sql_functions_in_query(s,**kwargs)

        
        df[index_name] = df.index
        df = pu.filter_away_columns_by_data_type(df)

        new_df = pandasql.sqldf(s, locals())
        #new_df = pu.delete_columns(new_df,index_name)
        if len(new_df[index_name]) == 0:
            return df_orig.iloc[[],:]
        return_df = df_orig.iloc[new_df[index_name],:]
    except:
        return_df = df_orig.query(query)
        
    return_df = pu.delete_columns(return_df,index_name)
    return return_df
    

def turn_off_scientific_notation(n_decimal_places=4):
    pd.set_option('display.float_format', lambda x: f'%.{n_decimal_places}f' % x)
    
def find_all_rows_with_nan(df,return_indexes=True,return_non_nan_rows=False):
    if return_indexes:
        if return_non_nan_rows:
            return np.where(~df.isna().any(axis=1))[0]
        else:
            return np.where(df.isna().any(axis=1))[0]
    else:
        if return_non_nan_rows:
            return df[~df.isna().any(axis=1)]
        else:
            return df[df.isna().any(axis=1)]
        
def find_all_rows_without_nan(df,return_indexes=True,):
    return find_all_rows_with_nan(
        df,
        return_indexes=return_indexes,
        return_non_nan_rows=True)
    
def filter_away_nan_rows(df):
    return df[~(df.isna().any(axis=1))]
    
from IPython.display import display
def display_df(df):
    display(df)
    
def dicts_to_dataframe(list_of_dicts):
    return pd.DataFrame.from_dict(list_of_dicts)

def rename_columns(df,columns_name_map,in_place = False,):
    return_value = df.rename(columns=columns_name_map,inplace = in_place)
    if in_place:
        return df
    else:
        return return_value

def unique_rows(df,columns=None):
    return df.drop_duplicates(subset = columns)

def delete_columns(df,columns_to_delete):
    columns_to_delete = nu.convert_to_array_like(columns_to_delete)
    columns_to_delete = [k for k in columns_to_delete if k in df.columns]
    if len(columns_to_delete) > 0:
        return df.drop(columns=columns_to_delete)
    else:
        return df
    
# ----------- 1/25 Additon: Used for helping with clustering ----------- #
import numpy_utils as nu
import matplotlib.pyplot as plt
import numpy as np

def divide_dataframe_by_column_value(
    df,
    column,
    return_names = True
    ):
    """
    Purpose: To divide up the dataframe into 
    multiple dataframes
    
    Ex: 
    divide_dataframe_by_column_value(non_error_cell,
                                column="cell_type_predicted")
    
    """
    column = nu.to_list(column)
    tables = []
    table_names = []
    for b,x in df.groupby(column):
        table_names.append(b)
        tables.append(x)
        
    if return_names:
        return tables,table_names
    else:
        return tables

def split_df_by_columns(df,columns,return_names = False):
    return divide_dataframe_by_column_value(df,columns,return_names = return_names)
    
def plot_histogram_of_differnt_tables_overlayed(tables_to_plot,
                          tables_labels,
                          columns=None,
                          fig_title=None,
                           fig_width=18.5,
                            fig_height = 10.5,
                            n_plots_per_row = 4,
                            n_bins=50,
                            alpha=0.4,
                                           density=False):
    """
    Purpose: Will take multiple
    tables that all have the same stats and to 
    overlay plot them
    
    Ex: plot_histogram_of_differnt_tables_overlayed(non_error_cell,"total",columns=stats_proj)
    
    """
    if not nu.is_array_like(tables_to_plot):
        tables_to_plot = [tables_to_plot]
        
    if not nu.is_array_like(tables_labels):
        tables_labels = [tables_labels]
    
    ex_table = tables_to_plot[0]
    
    if columns is None:
        columns = list(cell_df.columns)
        
    n_rows = int(np.ceil(len(columns)/n_plots_per_row))
    
    fig,axes = plt.subplots(n_rows,n_plots_per_row)
    fig.set_size_inches(fig_width, fig_height)
    fig.tight_layout()
    
    if not fig_title is None:
        fig.title(fig_title)

    
    for j,col_title in enumerate(columns):
        
        row = np.floor(j/4).astype("int")
        column = j - row*4
        ax = axes[row,column]
        ax.set_title(col_title)
        
        for curr_table,curr_table_name in zip(tables_to_plot,tables_labels):
            curr_data = curr_table[col_title].to_numpy()
            ax.hist(curr_data,bins=n_bins,label=curr_table_name,alpha=alpha,
                    density=density)
            
        ax.legend()
        
def plot_histograms_by_grouping(df,
                               column_for_grouping,
                               **kwargs):
    
    dfs,df_names = divide_dataframe_by_column_value(df,
                                column=column_for_grouping,
                                                   )
    
    plot_histogram_of_differnt_tables_overlayed(tables_to_plot=dfs,
                                               tables_labels=df_names,
                                               **kwargs)
def new_column_from_row_function(df,row_function):
    """
    Purpose: To create a new column where each 
    row element is a function of the other rows
    
    Ex:
    def synapse_double(row):
        return row["synapse_id"]*2

    new_column_from_row_function(proofreading_synapse_df,synapse_double)
    """
    return df.apply(lambda row: row_function(row), axis=1)

def new_column_from_dict_mapping(
    df,
    dict_map,
    column_name,
    default_value=None):
    """
    If want the same value to exist if not
    in the dictionary set default_value = "same"
    """
    def new_func(row):
        try:
            return dict_map[row[column_name]]
        except:
            if default_value == "same":
                return row[column_name]
            else:
                return default_value
        
    return new_column_from_row_function(df,new_func)


def reset_index(df):
    return df.reset_index(drop=True)
    

import numpy as np
import matplotlib.pyplot as plt
import six

def df_to_render_table(
    data, 
    col_width=3.0, 
    row_height=0.625, 
    font_size=14,
    header_color='#40466e', 
    row_colors=['#f1f1f2', 'w'],
    edge_color='w',
    bbox=[0, 0, 1, 1],
    header_columns=0,
    fontsize_header=None,
    ax=None,
    transpose=True,
    float_fmt = ".3f",
    **kwargs):
    """
    Purpose: To render a dataframe nicely that can be put in a figure
    
    Ex:
    df = pd.DataFrame()
    df['date'] = ['2016-04-01', '2016-04-02', '2016-04-03']
    df['calories'] = [2200, 2100, 1500]
    df['sleep hours'] = [2200, 2100, 1500]
    df['gym'] = [True, False, False]

    
    
    df_to_render_table(df, header_columns=0, col_width=2.0)
    """
    new_float_fmt = f'{{:,{float_fmt}}}'
    print(f"new_float_fmt = {new_float_fmt}")
    if float_fmt is not None:
        data = data.copy()
        data.update(data[pu.float_columns(data)].applymap(new_float_fmt.format))

    
    
    if transpose:
        data = data.transpose()
    
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if not transpose:
        mpl_table = ax.table(
            cellText=data.values, 
            bbox=bbox, 
            colLabels=data.columns, 
            **kwargs
        )
    else:
        mpl_table = ax.table(
            cellText=data.values, 
            bbox=bbox, 
            rowLabels=data.index, 
            **kwargs
        )

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w',fontsize=fontsize_header)
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax

def example_df_to_render_table(transpose=False):
    df = pd.DataFrame()
    df["index_name"] = ["stat 1","stat 2","stat 3"]
    df['date'] = ['2016-04-01', '2016-04-02', '2016-04-03']
    df['calories'] = [2200, 2100, 1500]
    df['sleep hours'] = [2200, 2100, 1500]
    df['gym'] = [True, False, False]

    return df_to_render_table(df, header_columns=0, col_width=2.0)

from pathlib import Path
def save_df(df,filename):
    filename = Path(filename)
    filename_str = str(filename.absolute())
    #print(f"{filename_str[-4]}") 
    if filename_str[-4:] != ".pkl":
        filename_str += ".pkl"
        
    df.to_pickle(filename_str)
    
def to_pickle(df,filename):
    return pu.save_df(df,filename)
    
def read_pickle(filepath):
    return pd.read_pickle(filepath)

def concat(df_list,**kwargs):
    return pd.concat(df_list,**kwargs)

from pathlib import Path
def df_to_csv(df,
              output_filename="df.csv",
              output_folder = "./",
              file_suffix = ".csv",
            output_filepath = None,
             verbose = False,
             return_filepath = True,
              compression = "infer",
              header = True,
             index = True,
             **kwargs):
    """
    Purpose: To export a dataframe as a csv
    file
    """
    if output_filepath is None:
        output_folder = Path(output_folder)
        output_filename = Path(output_filename)
        output_filepath = output_folder / output_filename

    output_filepath = Path(output_filepath)
    
    if str(output_filepath.suffix) != file_suffix:
            output_filepath = Path(str(output_filepath) + file_suffix)
    
    output_path = str(output_filepath.absolute())
    if verbose: 
        print(f"Output path: {output_path}")
        
    df.to_csv(str(output_filepath.absolute()), sep=',',index=index,compression=compression,header=header,**kwargs)
    
    if return_filepath:
        return output_path

def df_to_gzip(df,
              output_filename="df.gzip",
              output_folder = "./",
            output_filepath = None,
             verbose = False,
             return_filepath = True,
             index = False,):
    """
    Purpose: Save off a compressed version of dataframe
    (usually 1/3 of the size)
    
    """
    return df_to_csv(df,
              output_filename=output_filename,
              output_folder = output_folder,
              file_suffix = ".gzip",
            output_filepath = output_filepath,
             verbose = verbose,
             return_filepath = return_filepath,
              compression = "gzip",
             index = index,)

def gzip_to_df(filepath):
    return pd.read_csv(filepath,
                       compression='gzip', header=0, sep=',', quotechar='"', error_bad_lines=False)

    
import numpy as np
def filter_away_columns(df,
                       column_list):
    """
    Purpose: To filter away certain columns from
    a dataframe
    
    Ex: 
    
    filter_away_columns(proof_with_nucl,
                       ["id_x","pt_position_x"])
    """
    total_columns = list(df.columns)
    final_columns = np.setdiff1d(total_columns,column_list)
    return df[final_columns]


def csv_to_df(csv_filepath,remove_unnamed_columns=True):
    return_df = pd.read_csv(csv_filepath) 
    if remove_unnamed_columns:
        return_df = pu.remove_unnamed_columns(return_df)
    return return_df

def feather_to_df(path,**kwargs):
     return pd.read_feather(path,**kwargs)

read_feather = feather_to_df
    
def json_to_df(filepath,lines=True):
    return pd.read_json(filepath,lines=lines) 

def extract_arrays_from_df(df,column_names="closest_sk_coordinate",dtype="float",n_cols=3):
    #return np.array([np.array(k,dtype=dtype) for k in df["closest_sk_coordinate"].to_numpy()]).reshape(-1,n_cols)
    if nu.is_array_like(column_names):
        n_cols = len(column_names)
    return np.array([np.array(k,dtype=dtype) for k in df[column_names].to_numpy()]).reshape(-1,n_cols)

def extract_inividual_columns_from_df(df,column_names,dtype="float"):
    return extract_arrays_from_df(df,column_names=column_names,dtype=dtype).T

def df_to_list(df,return_tuples = False):
    vals = df.values.tolist()
    if return_tuples:
        vals = [tuple(k) for k in vals]
    return vals

def columns_values_from_df(df,columns,as_array=True):
    """
    Will return the values of columns as seperate variables 
    """
    return [df[k].to_list() if not as_array else df[k].to_numpy() for k in columns]

def datatypes_of_columns(df):
    return auto_proof_df.dtypes

def memory_usage_of_columns(df):
    return df.memory_usage(deep=True)

def add_prefix_to_columns(df,prefix):
    return df.add_prefix(prefix)

def df_from_dict_key_values_columns(data_dict,
                                   columns_names = ("attribute","description")):
    edge_df = pd.DataFrame.from_records([list(data_dict.keys()),
                          list(data_dict.values())]).transpose()
    edge_df.columns = columns_names
    return edge_df

def set_pd_max_row_col(max_rows = None,max_cols=None):
    pd.set_option('display.max_rows', max_rows)
    pd.set_option('display.max_columns', max_cols)
    
def set_pd_display_no_truncate():
    pu.set_pd_max_row_col()
def set_pd_display_no_truncate_col():
    pd.set_option('display.max_columns', None)
    
def reset_pd_display():
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')
    
from tqdm_utils import tqdm
def flatten_nested_dict_df_slow(df):
    """
    Purpose: Will faltten a dataframe if columns contain nested dicts
    """
    dicts = pu.dataframe_to_row_dicts(df)
    dicts_flattened = []
    for k in tqdm(dicts):
        curr_flat = pd.json_normalize(k, sep='_').to_dict(orient='records')[0]
        dicts_flattened.append(curr_flat)
    return pd.DataFrame.from_records(dicts_flattened)

import time
import json
def flatten_nested_dict_df(df,verbose = False):
    st = time.time()
    json_struct = json.loads(df.to_json(orient="records"))    
    df_flat = pd.io.json.json_normalize(json_struct) #use pd.io.json
    if verbose:
        print(f"flattening df time: {time.time() - st}")
    return df_flat


def col_to_numeric_array(col=None,
                        df=None,
                        col_name = None):
    """
    Purpose: to output a column as a numeric array
    (will account for coersion)
    
    """
    if col is None:
        col = df[col_name]
    x = pd.to_numeric(col,errors='coerce').to_list()
    return x

def convert_columns_to_numeric(df,columns):
    columns = nu.convert_to_array_like(columns)
    for k in columns:
        df[k] = pd.to_numeric(df[k])
        
    return df

from functools import reduce
def restrict_df_by_dict(df,dict_restriction,return_indexes=False):
    mask = reduce(lambda x,y: x & y, [df[k] == v for k,v in dict_restriction.items()])
    if return_indexes:
        return np.where(mask.to_numpy())[0]
    else:
        return df[mask]
    
def fillna(df,value=0,columns=None):
    if columns is None:
        columns = df.columns
    return df.fillna({k:value for k in columns})

def column_data_types(df):
    return df.dtypes
def filter_away_columns_by_data_type(df,
                                     data_type = "object",
                                    verbose = False):
    if data_type == "object":
        cols_to_inspect = df.columns[df.dtypes == "object"]
        cols_to_delete = []
        for c in cols_to_inspect:
            curr_col = df[c]
            curr_col = curr_col[~curr_col.isna()]

            if len(curr_col) > 0:
                if curr_col.infer_objects().dtypes == "object":
                    if verbose:
                        print(f"{c} should be filtered away")
                    cols_to_delete.append(c)
        return pu.delete_columns(df,cols_to_delete)

    else:
        return df.loc[:,df.columns[df.dtypes != data_type]]

import numpy_utils as nu
def duplicated_col_value_rows(
    df,
    col,
    ):
    """
    Ex:
    df_test = pd.DataFrame.from_records([
    dict(x=5,y=7,z=10),
    dict(x=5,y=7,z=12),
    dict(x=5,y=6,z=10)
    ])

    pu.duplicated_col_value_rows(df_test,col=["x","z"])
    
    """
    col = nu.convert_to_array_like(col)
    return df[df.duplicated(subset=col,keep=False)]

repeated_col_value_rows = duplicated_col_value_rows

def filter_away_rows_with_duplicated_col_value(
    df,
    col,
    ):
    
    col = nu.convert_to_array_like(col)
    return df[~df.duplicated(subset=col,keep=False)]


def filter_to_extrema_k_of_group(
    df,
    group_columns,
    sort_columns,
    k,
    extrema = "largest"):
    if extrema == 'largest':
        ascending = False
    elif extrema == 'smallest':
        ascending = True
        
    df_sort = pu.sort_df_by_column(
        df,sort_columns,
        ascending=ascending,
    )
    
    return df.groupby(group_columns).head(k)

def filter_to_first_k_of_group(
    df,
    group_columns,
    sort_columns,
    k,
    ):
    
    return filter_to_extrema_k_of_group(
    df,
    group_columns=group_columns,
    sort_columns=sort_columns,
    k=k,
    extrema = "largest").reset_index(drop=True)

def filter_to_first_instance_of_unique_column(
    df,
    column_name,
    reset_index = False,
    verbose = False,):
    column_name = nu.convert_to_array_like(column_name) 
    if verbose:
        print(f"Before filtering for unique {column_name}: {len(df)}")
    return_df = df.groupby(column_name).first()
    if verbose:
        print(f"AFTER filtering for unique {column_name}: {len(return_df)}")
    if reset_index:
        return_df = return_df.reset_index(drop=False)
        
    return return_df

def filter_df_rows_with_df(df,df_filter):
    keys = list(df_filter.columns.values)
    i1 = df.set_index(keys).index
    i2 = df_filter.set_index(keys).index
    return df[~i1.isin(i2)]

def combine_columns_as_str(
    df,
    columns,
    join_str = "_",
    ):
    
    return df[columns].astype('str').agg(join_str.join,axis=1)

import general_utils as gu
def filter_away_characters_from_column(
    df,
    column=None,
    characters_to_filter = ["_","/"],
    ):
    """
    Purpose: To filter out certain characters of a column
    values 
    """
    if column is None:
        column = list(df.columns)
    
    for c in column:
        def elim_func(row):
            if type(row[c]) == str:
                return gu.str_filter_away_characters(row[c],characters_to_filter)
            else:
                return row[c]

        df[c] = pu.new_column_from_row_function(
            df,
            elim_func)
    return df

def remove_unnamed_columns(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def normalize_df(
    df,
    column_means = None,
    column_stds = None,
    columns = None,
    in_place = False,
    ):
    """
    Purpose: To normalize a pandas table with
    means and standard deviations of the columns
    """
    if columns is None:
        columns = list(df.columns)
    if column_means is None:
        column_means = df[columns].mean(axis=0).to_numpy()

    if column_stds is None:
        column_stds = df[columns].std(axis=0).to_numpy()

    norm_columns = (df[columns]-column_means)/column_stds
    if in_place:
        df[columns] = norm_columns
        return df
    else:
        if len(columns) == len(df.columns):
            return norm_columns
        else:
            normalized_df = df.copy()
            normalized_df[columns]=norm_columns
            return normalized_df

def normalize_df_with_df(
    df,
    df_norm,
    verbose = False,
    ):
    """
    Purpose: To normalize a dataframe with
    another dataframe

    Pseudocode: 
    1) Get the columns of dataframe
    2) Use the columns of df to restrict the normalization df
    3) Get the column means and standard deviations
    """

    columns = list(df.columns)
    df_standardization = df_norm[columns]

    try:
        col_means = df_standardization.loc["norm_mean",:].to_numpy()
    except:
        col_means = df_standardization.iloc[0,:].to_numpy()

    try:
        col_stds = df_standardization.loc["norm_std",:].to_numpy()
    except:
        col_stds = df_standardization.iloc[1,:].to_numpy()

    if verbose:
        print(f"col_means = {col_means}")
        print(f"col_stds = {col_stds}")

    #raise Exception("")
    return pu.normalize_df(
        df,
        column_means = col_means,
        column_stds = col_stds,)

def normalize_df_with_names(df):
    col_means = df.mean(axis=0).to_numpy()
    col_stds = df.std(axis=0).to_numpy()
    df_standardization = pd.DataFrame(np.array([col_means,col_stds]),
         index=["norm_mean","norm_std"],
        columns=df.columns)
    return df_standardization

# Filter away rows with infinity
def filter_away_non_finite_rows(df,columns = None):
    return df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

def replace_None_with_default(df,default=0,columns=None):
    return pu.fillna(df,default,columns=columns)

def replace_nan_with_default(df,default=None):
    return df.where(pd.notnull(df), default)

def set_column_value_of_query(df,column,query,value):
    """
    Purpose: Set colum value of query


    """

    df.loc[df.query(query).index,column] = value
    return df



def group_by_func_to_column_old(
    df,
    columns,
    name = None,
    add_to_original_df = True,
    function_name = "count",
    ):
    dum_col= "xnusid"
    df[dum_col] = 1
    rename_df = getattr(df.groupby(columns),function_name)()[[dum_col]]
    if name is None:
        name = "_".join(columns)
        name = f"{function_name}_{name}"

    rename_df = pu.rename_columns(rename_df,{dum_col:name})
    

    if add_to_original_df:
        rename_df = pd.merge(df,rename_df,on=columns,how="left")

    if dum_col in rename_df.columns:
        rename_df = pu.delete_columns(rename_df,[dum_col])
        
    return rename_df

def group_by_func_to_column_slow(
    df,
    columns,
    name = None,
    add_to_original_df = True,
    function_name = "count",
    ):
    dum_col= "xnusid"
    df[dum_col] = 1
    rename_df = getattr(df.groupby(columns),function_name)()[[dum_col]]
    if name is None:
        name = "_".join(columns)
        name = f"{function_name}_{name}"

    rename_df = pu.rename_columns(rename_df,{dum_col:name})
    

    if add_to_original_df:
        rename_df = pd.merge(df,rename_df,on=columns,how="left")

    if dum_col in rename_df.columns:
        rename_df = pu.delete_columns(rename_df,[dum_col])
        
    return rename_df

def group_by_func_to_column(
    df,
    columns,
    name = None,
    add_to_original_df = True,
    function_name = "count",
    ):
    dum_col= "xnusid"
    df[dum_col] = 1
    rename_df = df[columns + [dum_col]].groupby(columns).transform(function_name)
    
    if name is None:
        name = "_".join(columns)
        name = f"{function_name}_{name}"

    rename_df = pu.rename_columns(rename_df,{dum_col:name})
    

    if add_to_original_df:
        df[name] = rename_df[name]
        rename_df = df

    if dum_col in rename_df.columns:
        rename_df = pu.delete_columns(rename_df,[dum_col])
        
    return rename_df


def drop_rows_with_all_nans(df):
    return df.dropna(how="all")

def sample_rows(df,n_samples,seed=None):
    if len(df) == 0:
        return df
    if len(df) <= n_samples:
        n_samples = len(df)
    return df.sample(n_samples,random_state=seed)

def sample_columns(df,n_samples):
    return df.sample(n_samples, axis=1)

def expand_array_column_to_separate_columns(
    df,
    columns,
    use_previous_column_name = True,
    ):
    """
    Will expand a column that has an array stored in each
    row to moving the values to each with its own column
    
    Ex: 
    import pandas_utils as pu
    pu.expand_array_column_to_separate_columns(
    use_previous_column_name = False,
    columns = "embedding",
    df = df_old,
    )
    """
    for column in nu.convert_to_array_like(columns):

        sub_df = pd.DataFrame(np.vstack(df[column].to_numpy()))
        if use_previous_column_name:
            column_names = [f"{column}_{i}" for i in range(0,len(sub_df.columns))]
            sub_df.columns = column_names

        df = pd.concat([df,sub_df],axis=1)
        df = pu.delete_columns(df,column)


    return df

def sort_df_by_column(
    df,
    columns,
    ignore_index = True,
    ascending = False,
    na_position = "last" #where to put the nans
    ):
    columns = nu.convert_to_array_like(columns)
    return df.sort_values(by=columns,
                         ascending=ascending,
                          ignore_index=ignore_index,
                         na_position = na_position)


def histogram_of_discrete_labels(
    df,
    y,
    x = "index",
    x_steps = None,
    step_size = None,
    nbins = 100,
    verbose = False,
    cumulative = False,
    normalize = True,
    add_counts = True,
    plot = False,
    max_percentile = 99,
    **kwargs,
    ):
    """
    Purpose: To create a cumulative or discrete
    df of discrete values accordinate to another
    indexing value

    Ex: 
    pu.histogram_of_discrete_labels(
        df_centr_sort,
        y="cell_type",
        normalize=False,
        cumulative=False
    )
    
    Ex 2: 
    import cell_type_utils as ctu
    df_counts = pu.histogram_of_discrete_labels(
        df_centr_sort,
        y="cell_type_fine",
        normalize=True,
        cumulative=True,
        x_steps = np.linspace(0,1000,100),
        plot = True,
        color_dict = ctu.cell_type_fine_color_map,
        set_legend_outside_plot = True,

        )
    """
    
    
    x_range = x_steps
    
    df = df.query(f"{y}=={y}").reset_index(drop = False)
    all_ys = df[y].unique()
    all_ys = all_ys[all_ys != "index"]
    
    
    x_vals = getattr(df,x)
    
    if max_percentile is not None:
        max_value  = np.percentile(x_vals,max_percentile)
        if max_value <= np.min(x_vals):
            max_value = np.max(x_vals)
    else:
        max_value = np.max(x_vals)
    
    if verbose:
        print(f"All labels = {all_ys}")
        
    if x_range is None:
        if step_size is None:
            x_range = np.linspace(np.min(x_vals),max_value,nbins)
        else:
            x_range = np.arange(np.min(x_vals),max_value+0.1,step_size)
    
    
    
    counts_records = []
#     print(f"len(df) = {len(df)}")
#     print(f"x_range = {x_range}")
    for j,x_val in enumerate(x_range[1:]):
        x_lower = x_range[j]
        x_middle = np.mean([x_val,x_lower])
        #print(f"x_middle = {x_middle}")

        local_dict = dict(x = x_middle)

        if cumulative:
            x_lower = np.min(x_range)

        df_curr = df.query(f"({x} >= {x_lower}) and ({x} <= {x_val})")
        #print(f"df_curr = {len(df_curr)}")

        total_counts = len(df_curr)
        if total_counts == 0:
            continue
        for lab in all_ys:
            curr_counts = df_curr.query(f"{y} == '{lab}'")
            if normalize:
                local_dict[lab] = len(curr_counts)  / total_counts
            else:
                local_dict[lab] = len(curr_counts)
        
        if add_counts:
            local_dict["counts"] = total_counts
            
        #print(f"total_counts = {total_counts}")

        counts_records.append(local_dict)

    df_counts = pd.DataFrame.from_records(counts_records)
    #print(f"df_counts = {len(df_counts)}")
    
    if plot:
        import matplotlib_utils as mu
        mu.stacked_bar_graph(
           df_counts,
            verbose = verbose,
            **kwargs
        )
    return df_counts

def unique_row_counts(
    df,
    columns=None,
    count_column_name = "unique_counts",
    add_to_df = False,
    verbose = False,
    ):

    """
    Purpose: To determine the counts of the 
    number of unique rows as determined by the columns
    (and could add the number to the dataframe)
    """
    unique_df = df
    if columns is None:
        columns = list(df.columns)

    columns = nu.convert_to_array_like(columns)

    count_column = count_column_name
    unique_df[count_column] = 1
    counts_df = unique_df[columns + [count_column]].groupby(columns).count().reset_index()
    if verbose:
        print(f"# of unique rows = {len(counts_df)}")
    
    if add_to_df:
        df = pd.merge(
            pu.delete_columns(df,count_column),
            counts_df,
            on=columns,
            how="left")
        return df
    else:
        return counts_df
    
def set_max_colwidth(width=400):
    pd.set_option('max_colwidth', width)
    
import numpy_utils as nu
def filter_df_by_column_percentile(
    df,
    columns=None,
    percentile_buffer=1,
    percentile_lower = None,
    percentile_upper = None,
    verbose = False,
    ):
    """
    Purpose: To filter a dataframe for
    a certain percentile range
    
    Ex: 
    pu.filter_df_by_column_percentile(
        df_to_plot,
        "n_syn_soma_no_label_postsyn",
        percentile_buffer = 5,
        verbose = True
    )
    """
    if columns is None:
        columns= list(df.columns)
    columns = nu.convert_to_array_like(columns)
    if percentile_lower is None:
        if percentile_buffer is not None:
            percentile_lower = percentile_buffer
        else: 
            percentile_lower = 0
    
    if percentile_upper is None:
        if percentile_buffer is not None:
            percentile_upper = 100 - percentile_buffer
        else:
            percentile_upper = 100
            
    
    for attribute in columns:
        att_vals = df.query(f"{attribute} == {attribute}")[attribute].to_numpy()
        min_value = np.percentile(att_vals,percentile_lower)
        max_value = np.percentile(att_vals,percentile_upper)
        if verbose:
            print(f"outlier processing: min_value = {min_value}, max_value = {max_value}")

        original_size = len(df)
        df = df.query(
            f"({attribute}>={min_value})"
            f" and ({attribute}<={max_value})"
        )

        if verbose:
            print(f"After filtering df by column {attribute}: reduced from {original_size} to {len(df)} entries")
            
    return df

percentile_filter = filter_df_by_column_percentile

def filter_away_rows_with_nan_in_columns(
    df,
    columns,
    verbose = False,):
    """
    Purpose: To filter away rows that have
    a nan value in certain columns

    """
    try:
        df = df.reset_index()
    except:
        pass
    
    columns = nu.convert_to_array_like(columns)
    
    df_curr = df[columns]
    idx = pu.find_all_rows_without_nan(
        df_curr,return_indexes = True,
    )
    df_filt = df.iloc[idx,:]
    
    if verbose:
        print(f"Filtering {len(df)} rows to {len(df_filt)} rows")
    
    for c in columns:
        df_filt = df_filt.query(f"{c}=={c}")
        
    try:
        df_filt.reset_index()
    except:
        pass
    
    return df_filt

def filter_away_rows_with_nan_only_in_columns(
    df,
    columns,
    verbose = False,):
    if verbose:
        print(f"Before nan filtering in columns: {len(df)}")
    
    for c in columns:
        df = df.query(f"{c}=={c}")
        
    try:
        df = df.reset_index(drop=True)
    except:
        pass
    
    if verbose:
        print(f"-> After nan filtering in columns: {len(df)}")
    
    return df
    
def filter_away_rows_with_inf_only_in_columns(
    df,
    columns,
    in_place = False,
    verbose = False,):
    
    if verbose:
        print(f"Before inf filtering in columns: {len(df)}")
    if not in_place:
        df = df.copy()
        
    df_filt = df[columns].replace([np.inf, -np.inf], np.nan)
    idx = pu.find_all_rows_without_nan(df_filt)
    df_filt = df.iloc[idx,:]
    try:
        df_filt = df_filt.reset_index(drop=True)
    except:
        pass
    
    if verbose:
        print(f"-> After inf filtering in columns: {len(df_filt)}")
    
    return df_filt

def filter_away_rows_with_inf_or_nan_only_in_columns(
    df,
    columns,
    verbose = False,):
    df = pu.filter_away_rows_with_nan_only_in_columns(
        df,
        columns=columns,
        verbose=verbose,
    )
    
    df = pu.filter_away_rows_with_inf_only_in_columns(
        df,
        columns=columns,
        verbose=verbose,
    )
    
    return df

def randomly_sample_df(
    df,
    n_samples=None,
    seed = None,
    replace = False,
    verbose = False,):
    
    #print(f"seed = {seed}")
    if seed is not None:
        np.random.seed(seed)
        
    if n_samples is None:
        n_samples = len(df)

    if (n_samples > len(df)) and replace == False:
        if verbose:
            print(f"# of samples greater than df length so just returning df")
        return df
    idx = nu.randomly_sample_array(np.arange(len(df)),n_samples,replace = replace)
    restricted_df = df.iloc[idx,:].reset_index(drop = True)
    return restricted_df

def shuffle_df(df,seed=None,**kwargs):
    return pu.randomly_sample_df(
        df,
        seed=seed,
        **kwargs)

import general_utils as gu
def expand_array_column_to_separate_rows(
    df,
    array_columns,
    scalar_columns = None,
    verbose = False,
    ):
    
    """
    Purpose: to take columns that have arrays stored in them and turn them
    into rows, add on the scalar rows if requested
    
    Ex:
    import pandas as pd
    array_columns = ["spine_intervals_1","spine_intervals_2"]
    scalar_columns = ["compartment","n_spine","segment_id","width"]

    out_df = pu.expand_array_column_to_separate_rows(
        df = inter_attr_df,
        array_columns=array_columns,
        scalar_columns=scalar_columns
    )
    """
    
    array_columns= nu.convert_to_array_like(array_columns)

    curr_data = {k:df[k].to_list() for k in array_columns}
    dtype_map = {}

    if scalar_columns is not None:
        first_key = list(curr_data.keys())[0]
        scalar_columns = nu.convert_to_array_like(scalar_columns)
        for s in scalar_columns:
            curr_data[s] = [[k]*len(v) for k,v in zip(df[s].to_list(),curr_data[first_key])]
            dtype_map[s] = df[s].dtype
            
        #return curr_data

    curr_data = dict([(k,gu.combine_list_of_lists(v)) if len(v) > 0 else
                      (k,v) for k,v in curr_data.items()])
    df = pd.DataFrame.from_dict(curr_data)
    df = pu.set_column_datatypes(df,dtype_map)
    return df

def set_column_datatypes(
    df,
    datatype_map):
    
    return df.astype(datatype_map)
    
def is_column_numeric(df,column):
    if(df[column].dtype == np.float64 or df[column].dtype == np.int64):
        return True
    else:
        return False
    
def normalize_rows_by_sum(
    df,
    fillna_value=0,
    percentage = False
    ):
    """
    Purpose: To divide the rows of a dataframe by the sum  
    of the rows
    """

    if fillna_value is not None:
        df = pu.fillna(df,fillna_value)
    
    divisor = df.sum(axis=1)
    if percentage:
        divisor = divisor/100
        
    return df.div(divisor, axis=0)


def concat_dfs_without_duplicates(
    dfs,
    columns=None,
    keep = "first",
    verbose = False
    ):
    """
    Purpose: Want to do a concatenate on a list as long as 
    there are no repeats in certain columns beforehand

    Pseudocode: 
    1)concatenate the dataframes and reset the index
    2) drop the duplicates (after restricting to certain columns)
    3) Get the indexes of the dropped duplicates and use that to restrict the concatenated dataframe
    4) Reset the index and return
    """
    df_combined = pd.concat(dfs).reset_index(drop=True)
    
    if columns is None:
        columns = list(df_combined.columns)

    if "str" in str(type(columns)):
        columns = [columns]

    unique_df = df_combined[columns].drop_duplicates(keep="first")
    final_df = df_combined.iloc[unique_df.index,:].reset_index(drop=True)

    if verbose:
        print(f"# of duplicate rows = {len(df_combined) - len(final_df)}")
    return final_df

def is_dataframe(df):
    return isinstance(df, pd.DataFrame)
def is_series(df):
    return isinstance(df, pd.Series)

import numpy_utils as nu
import pandas_utils as pu

def summary_statistic_over_columns(
    df,
    columns,
    summary_statistic="mean",
    weight = None,
    summary_statisic_args = None,
    
    #for the naming
    prefix = None,
    suffix = None,
    append_statistic_name = False,
    special_count_name = False,
    
    # for outputting
    verbose = False,
    return_df = False,
    debug_time = False
    ):
    """
    Purpose: To compute a summary statistic for columns 
    over all rows of dataframe
    
    Ex: 
    import pandas_utils as pu
    pu.summary_statistic_over_columns(
        columns = ["n_synapses"],#["","n_synapses_pre"],
        df = node_df,
        summary_statistic = "percentile",
        summary_statisic_args = 99,
        verbose = True,
        return_df = False,
    append_statistic_name = True
    )

    """
    st = time.time()
    
    singular_flag = False
    
    if columns is None:
        default_column = list(df.columns)[0]
        columns = [default_column]
    else:
        default_column = None
        
        
    if not nu.is_array_like(columns):
        singular_flag = True
        columns = [columns]
        
    columns = list(columns)
    #print(f"columns before reduction = {columns}")
    # only keep the columns that are in df
    columns = [k for k in columns if k in df.columns]
    #print(f"   -> columns after reduction = {columns}")
    
    node_df_restr = df[columns]

    if summary_statisic_args is not None:
        summary_statisic_args = nu.convert_to_array_like(summary_statisic_args)
    else:
        summary_statisic_args = []

    if summary_statistic == "percentile":
        summary_statistic = "quantile"
        if summary_statisic_args is not None:
            summary_statisic_args = [k/100 if type(k) in [int,float]
                                    else k for k in summary_statisic_args]

    if verbose:
        print(f"summary_statistic = {summary_statistic}\n   -> summary_statisic_args = {summary_statisic_args}")
    
    if weight is not None and summary_statistic == "mean":
        summary_df = pd.Series(data=[nu.average_by_weights(node_df_restr[k].to_numpy(),
                                                          weights = df[weight].to_numpy())
                                    for k in columns],
                               index = columns)
    else:
        summary_df = getattr(node_df_restr,summary_statistic)(*summary_statisic_args)
        
    if debug_time:
        print(f"Time for computing summary statistic = {time.time() - st}")
        st = time.time()

    
    if suffix is not None:
        summary_df.index = [f"{k}{suffix}" for k in summary_df.index]
    
    if prefix is not None:
        summary_df.index = [f"{prefix}_{k}" for k in summary_df.index]
        
    if append_statistic_name:
        if summary_statistic == "count" and special_count_name:
            if prefix is not None:
                summary_df.index = [k.replace(prefix,f"n_{prefix}s") for k in summary_df.index]
            else:
                summary_df.index = [f"n_{k}" for k in summary_df.index]
        else:
            summary_df.index=[f"{k}_{summary_statistic}{''.join([f'_{j}' for j in summary_statisic_args])}" 
                              for k in summary_df.index]
            
    if default_column is not None:
        summary_df.index = [k.replace(f"{default_column}_","") if k !=
                            default_column else k for k in summary_df.index]
        summary_df.index = [k.replace(default_column,"") if k !=
                            default_column else k for k in summary_df.index]
        summary_df.index = [k[:-1] if k[-1] == "_"
                     else k for k in summary_df.index]
        
    if summary_statistic == "count":
        summary_df = summary_df.astype('int')
        
    if debug_time:
        print(f"Fixing the names = {time.time() - st}")
        st = time.time()
        
            
    if not return_df:
        return_dicts = pu.df_to_dicts(summary_df)
        if singular_flag:
            return list(return_dicts.values())[0]
        else:
            return return_dicts
    else:
        return summary_df
    
    
import numpy_utils as nu
def summary_statistics_over_columns_by_category(
    df,
    category_columns,
    prefix = None,
    attribute_summary_dicts = None,
    add_counts_summary = False,
    special_count_name = True,
    verbose = False,
    debug_time = False,
    ):
    """
    summary_statistics_over_columns_by_category()

    def 

    Purpose: Want to compute a bunch 
    of category counts/or statistic
    of certain columns

    Pseudocode: 
    1) Create a dataframe of the attribute
    2) Get all the possible combinations of restrictions for categories
    3) 
    
    
    Ex: 
    node_dict = G.nodes["L1_2"]
    attribute = "synapse_data"
    prefix = "synapse"
    df = pd.DataFrame.from_records(node_dict[attribute])

    pu.summary_statistics_over_columns_by_category(
        df,
        prefix = "synapse",
        category_columns = ["head_neck_shaft","syn_type"],
        attribute_summary_dicts = [dict(columns="volume",
                                       summary_statistic = "sum",
                                        summary_statisic_args = None)],
        add_counts_summary = True,
        verbose = False,
        special_count_name = True,

    )
    """
    st = time.time()

    if attribute_summary_dicts is None:
        attribute_summary_dicts = []

    if type(attribute_summary_dicts) == dict:
        attribute_summary_dicts = [attribute_summary_dicts]

    if add_counts_summary:
        attribute_summary_dicts.append(
            dict(columns = None,summary_statistic = "count")
        )


    category_combinations = [None]
    if category_columns is not None:
        category_columns_revised = []
        for k in category_columns:
            if k in df.columns:
                category_columns_revised.append(k)
        else:
            if verbose:
                print(f"No column {k} for category")
        
        if len(category_columns_revised) > 0:
            category_columns = category_columns_revised
            category_columns = nu.convert_to_array_like(category_columns)
            category_attributes_pos = [list(df[k].unique()) + [None]
                                      for k in category_columns]
            category_combinations = nu.all_combinations_of_lists(*category_attributes_pos)
            if verbose:
                print(f"category_combinations = {category_combinations}")
        else:
            category_columns = [None]
            
    if debug_time:
        print(f"Time for combinations = {time.time() - st}")
        st = time.time()
    

    results_dict = dict()
    for curr_restr in category_combinations:
        if verbose:
            print(f"\n-- Working on curr_restr = {curr_restr}--")
        suffix = ""
        if curr_restr is None:
            curr_df = df
        else:
            queries = []
            for k,v in zip(category_columns,curr_restr):
                if v is not None:
                    suffix += f"_{v}"
                    queries.append(f"({k} == '{v}')")

            if len(queries) == 0:
                curr_df = df
            else:
                total_query = " and ".join(queries)
                if verbose:
                    print(f"total_query = {total_query}")
                curr_df = df.query(total_query)

        if verbose:
            print(f"  len(df) {suffix} = {len(curr_df)}")
            
        if debug_time:
            print(f"Time for query = {time.time() - st}")
            st = time.time()

        # compute the statistics over the current dataframe
        for stat_dict in attribute_summary_dicts:
            summary_statisic_args = stat_dict.get("summary_statisic_args",None)
            summary_statistic = stat_dict["summary_statistic"]
            columns= stat_dict["columns"]

            if columns is not None:
                columns = nu.convert_to_array_like(columns)

            results = pu.summary_statistic_over_columns(
                columns = columns,#["","n_synapses_pre"],
                df = curr_df,
                summary_statistic = summary_statistic,
                summary_statisic_args = summary_statisic_args,
                verbose = False,
                return_df = False,
                append_statistic_name = True,
                suffix = suffix,
                prefix = prefix,
                special_count_name=special_count_name,
                debug_time=debug_time
            )
            
            if debug_time:
                print(f"Time for summary_statistic_over_columns = {time.time() - st}")
                st = time.time()

            results_dict.update(results)
    return results_dict

def example_Series():
    return pd.Series(data=[2255,81],index=["n_synapses",'n_synapses_pre'])

def df_from_array(
    array,
    columns,
    inf_fill_value = 10000,
    ):
    
    df_temp = pd.DataFrame(array)
    df_temp.columns = columns
    if inf_fill_value is not None:
        df_temp=df_temp.replace([np.inf],inf_fill_value)
    return df_temp

def filter_columns(
    df,
    columns_to_keep = None,
    columns_to_delete = None):
    """
    Purpose: To filter columns
    of dataframe for those specified to keep
    and those specified to delete
    
    Ex: 
    pu.filter_columns(
        normalization_df,
        #columns_to_keep = ["skeletal_length",'skeleton_vector_upstream_theta'],
        columns_to_delete=["skeletal_length"]
    )
    """
    
    if columns_to_keep is not None:
        columns_to_keep = list(nu.convert_to_array_like(columns_to_keep))
        df = df[columns_to_keep]
        
    if columns_to_delete is not None:
        columns_to_delete = list(nu.convert_to_array_like(columns_to_delete))
        df = pu.delete_columns(df,columns_to_delete)
        
    return df


def set_column_subset_value_by_query(
    df,
    query,
    column,
    value=None,
    column_for_value = None,
    function_for_value = None,
    verbose = False):
    """
    Purpose: Set column
    """
    
    curr_map = df.eval(query)
    if verbose:
        print(f"Number of rows in query = {curr_map.sum()}")
        
    if value is not None:
        df.loc[curr_map,column] = value
    elif function_for_value is not None:
        df.loc[curr_map,column] = function_for_value(
            df = df,
            idx_map = curr_map,
        )
    elif column_for_value is not None:
        df.loc[curr_map,column] = df.loc[curr_map,column_for_value]
    else:
        df.loc[curr_map,column] = value
    
    return df

def count_unique_column_values(
    df,
    column,
    sort = False,
    count_column_name = "unique_counts",
    add_percentage = False,):
    column = list(nu.convert_to_array_like(column))
    return_df =  pu.unique_row_counts(df[column],count_column_name=count_column_name)
    if sort:
        return_df = pu.sort_df_by_column(return_df,count_column_name)
        
    if add_percentage:
        return_df = pu.add_percentage_column(return_df,column = count_column_name)
    return return_df

def add_percentage_column(
    df,
    column,
    percentage_column = None,# "percentage"
    ):
    
    if percentage_column is None:
        percentage_column = f"percentage ({column})"
        
    df[percentage_column] = 100*df[column]/df[column].sum()
    return df
    

import numpy_utils as nu
def intersect_columns(dfs):
    return list(nu.intersect1d_multi_list([list(df.columns) for df in dfs]))

def intersect_df(
    df,
    df_restr,
    restriction_columns = None,
    append_restr_columns = True,
    reset_index = False,
    verbose = False,
    ):
    """
    Purpose: To get the intersection of two dataframes
    """
    
    if restriction_columns is None:
        restriction_columns = pu.intersect_columns([df,df_restr])
        
    if verbose:
        print(f"restriction_columns= {restriction_columns}")
        
    if append_restr_columns:
        cols = list(df_restr.columns)
    else:
        cols = restriction_columns
        
    restr_df = df.reset_index().merge(
        df_restr[cols],
        how="inner",
        on=restriction_columns,
    ).set_index("index")
    
    if reset_index:
        restr_df = restr_df.reset_index(drop=True)
        
    if verbose:
        print(f"Length of df after restriction using ({restriction_columns}): {len(restr_df)}")
        
    return restr_df

def setdiff_df(
    df,
    df_restr,
    restriction_columns = None,
    reset_index = False,
    verbose = False,
    inter_df = None,
    ):
    """
    Purpose: To find subtract the 
    rows of one dataframe from the other
    """
    if restriction_columns is None:
        restriction_columns = pu.intersect_columns([df,df_restr])
        
    if inter_df is None:
        inter_df = intersect_df(
            df,
            df_restr,
            restriction_columns = restriction_columns,
            append_restr_columns = False,
            reset_index = False,
            verbose = False,
            )
    
    diff_df = df.iloc[list(np.setdiff1d(
        df.index.to_list(),
        inter_df.index.to_list())
        ),:]
    
    if verbose:
        print(f"Length of df after restriction using ({restriction_columns}): {len(diff_df)}")
        
    if reset_index:
        diff_df = diff_df.reset_index(drop=True)
        
    return diff_df

def split_df_from_intersect_df(
    df,
    df_restr,
    restriction_columns = None,
    append_restr_columns = False,
    reset_index = True,
    verbose = False):
    """
    Purpose: To Split a table using a restriction df
    """
    inter_df = pu.intersect_df(
        df,
        df_restr,
        restriction_columns = restriction_columns,
        append_restr_columns = append_restr_columns,
        reset_index = False,
        verbose = False,
        )

    
    diff_df = pu.setdiff_df(
        df,
        df_restr,
        restriction_columns = restriction_columns,
        reset_index = reset_index,
        verbose = False,
        inter_df = inter_df,
        )
    
    if reset_index:
        inter_df = inter_df.reset_index(drop=True)
    
    if verbose:
        print(f"inter_df size = {len(inter_df)}, diff_df = {len(diff_df)}")
        
    return inter_df,diff_df
    
def split_str_column_into_multple(df,column,delimiter="_"):
    return df[column].str.split(delimiter,expand =True)

def excel_to_df(filename):
    return pd.read_excel(filename, index_col=0) 

def bin_df_by_column(
    df,
    column,
    bins=None,
    n_bins = 10,
    verbose = False,
    return_bins = False,
    equal_depth_bins = False,
    percentile_upper = None,
    ):
    """
    Purpose: To calculate sub dataframe
    after binning a certain column
    """
    if percentile_upper is not None:
        df = pu.filter_df_by_column_percentile(
            df,
            columns=column,
            percentile_lower=0,
            percentile_upper=percentile_upper,
            verbose=False,
        )
    

    if bins is None:
        if equal_depth_bins:
            bins = nu.equal_depth_bins(
                df[column].to_numpy(),
                n_bins = n_bins,
            )
        else:
            #bins = np.linspace(0,df[column].max(),n_bins + 1)
            bins = nu.equal_width_bins(
                df[column].to_numpy(),
                n_bins = n_bins,
            )

    if verbose:
        print(f"bins = {bins}")

    df_list = []
    for i in range(1,len(bins)):
        if i == len(bins) -1:
            final_comp = "<="
        else:
            final_comp = "<"

        query_str = f"({column} >= {bins[i-1]}) and ({column} {final_comp} {bins[i]})"
        curr_df = df.query(query_str)

        if verbose:
            print(f"For bin {i}, query_str = {query_str}")
            print(f"   -> length of sub df = {len(curr_df)}")
        df_list.append(curr_df)

    if return_bins:
        return df_list,bins
    return df_list

def bin_df_by_column_stat(
    df,
    column,
    func,
    bins=None,
    equal_depth_bins = False,
    percentile_upper = None,
    n_bins = 10,
    verbose = False,
    return_bins = True,
    return_bins_mid = False,
    return_df_len = True,
    return_std = True,
    func_std = None,
    plot = False,
    plot_n_data_points = True,
    plot_errorbars = False,
    return_plot = False,
    ):
    """
    Purpose: to compute a statistic over
    binned sub dfs (where bins are determined by column)
    """
    if plot:
        twin_color = "blue"
        fontsize = 20

        figsize = (10,5)

    df_bins,bins = pu.bin_df_by_column(
        df=df,
        column=column,
        bins=bins,
        n_bins = n_bins,
        verbose = verbose,
        return_bins = True,
        equal_depth_bins=equal_depth_bins,
        percentile_upper=percentile_upper,
        )
    
    if type(func) == str:
        curr_values = [np.array(k[func]).astype('float') for k in df_bins]
        df_stats = [k.mean() for k in curr_values]
        df_std = [k.std() for k in curr_values]
    else:
        df_stats = [func(k) for k in df_bins]
        if func_std is not None:
            df_std = [func_std(k) for k in df_bins]
        else:
            df_std = None
    df_len = [len(k) for k in df_bins]
    
    
    if plot:
        mid_bins = (bins[1:] + bins[:-1])/2

        fig,ax = plt.subplots(1,1,figsize=figsize)
        
        #ax.plot(mid_bins,df_stats,)
        if df_std is None or not plot_errorbars:
            ax.plot(mid_bins,df_stats,)
        else:
            ax.errorbar(mid_bins,df_stats,yerr = df_std)
        ax.set_xlabel(f"{column}")
        if type(func) == str:
            ax.set_ylabel(f"{func}")
        else:
            ax.set_ylabel(f"{func.__name__}")

        if plot_n_data_points: #and not equal_depth_bins:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('# of Data Points',fontsize=fontsize,color=twin_color)  # we already handled the x-label with ax1
            ax2.tick_params(axis='y', labelcolor=twin_color)
            ax2.plot(mid_bins, df_len, color=twin_color)
        #plt.show()
        
        if return_plot:
            return ax

    if (not return_bins) and (not return_df_len) and (not return_std):
        return df_stats
    return_list = [df_stats]
    
    if return_bins_mid:
        mid_bins = (bins[1:] + bins[:-1])/2
        return_list.append(mid_bins)
    elif return_bins:
        return_list.append(bins)
    if return_df_len:
        return_list.append(df_len)
        
    if return_std:
        return_list.append(df_std)
        
    return return_list

def empty_df(columns=None):
    """
    Purpose: Create an empty dataframe
    with certain columns
    """
    if columns is None:
        columns = []
        
    return pd.DataFrame.from_dict({k:[] for k in columns})



import numpy_utils as nu
def merge_df_to_source_target(
    df,
    df_append,
    source_name = "source",
    target_name = "target",
    append_type = "prefix",
    on = None,
    how = "left",
    columns = None,
    in_place = False,
    ):

    """
    Purpose: merge a dataframe to 
    a dataframe with a directional column
    """
    if columns is None:
        columns = list(df_append.columns)

    if not in_place:
        #print(f"Trying to copy")
        df = df.copy(deep=True)

    for k in [source_name,target_name]:
#         if append_type == "prefix":
#             name_str = 'f"{kk}_{v}"'
#         elif append_type == "suffix":
#             name_str = 'f"{v}_{kk}"'
#         elif append_type is None:
#             name_str = 'f"{kk}"'
#         else:
#             raise Exception("")

        name_str = None
        if k is not None and len(k) > 0:
            if append_type == "prefix":
                name_str = 'f"{kk}_{v}"'
            elif append_type == "suffix":
                name_str = 'f"{v}_{kk}"'
            elif append_type is None:
                name_str = 'f"{kk}"'
        
        
        if name_str is None:
            name_str = 'f"{v}"'
            #raise Exception("")

        
        rename_dict = {v:eval(name_str) for kk,v in zip([k]*len(columns),columns)}
        
        
        if on is None:
            curr_on = [v for v in rename_dict.values() if v in df.columns]
        else:
            if len(np.intersect1d(nu.convert_to_array_like(on),list(rename_dict.values()))) == 0:
                curr_on = k
                rename_dict[on] = k
                
        #print(f"rename_dict = {rename_dict}")
        try:
            return_value = pd.merge(
                df,
                pu.rename_columns(df_append[columns],
                                  rename_dict,
                                 in_place = in_place),
                on=curr_on,
                how=how,
                copy = not in_place,
            )
        except:
            rename_dict.update({v:eval(name_str) for kk,v in zip([k],[on])})
            curr_on = rename_dict[on]
            
            return_value = pd.merge(
                df,
                pu.rename_columns(df_append[columns],
                                  rename_dict,
                                      in_place = in_place),
                on=curr_on,
                how=how,
                copy = not in_place,
            )
            
        
        if not in_place or return_value is not None:
            df = return_value
        #print(f"df.columns = {df.columns}")
        #print(f"return_value.columns = {return_value.columns}")
            
    return df

append_df_to_source_target = merge_df_to_source_target

def df_to_index_dict(
    df,
    column = None,
    columns_to_export = None):
    """
    Purpose: To convert a pandas dataframe
    to just one dictionary indexed by the index
    """

    if column is not None:
        df = df.set_index(column)
        
    keys = df.index.to_list()
    
    if columns_to_export is None:
        columns_to_export = list(df.columns)
    values = df[columns_to_export].to_numpy()
    my_dict = {k:v for k,v in zip(keys,values)}
    return my_dict

def coordinate_columns(
    name="synapse",
    suffix = "nm",
    axes = ("x","y","z"),
    verbose = False,
    ):
    
    if suffix is not None and len(suffix) > 0:
        suffix = f"_{suffix}"
    else:
        suffix = ""

    if name is not None:
        prefix = f"{name}_"
    else:
        prefix = ""

    columns = [f"{prefix}{a}{suffix}" for
        a in axes]
    
    if verbose:
        print(f"columns = {columns}")
    
    return columns

def coordinates_from_df(
    df,
    name="synapse",
    suffix = "nm",
    axes = ("x","y","z"),
    filter_away_nans = False,
    columns = None,
    ):
    
    if columns is None:
        columns = coordinate_columns(
            name=name,
            suffix = suffix,
            axes = axes,
        )
    
    if filter_away_nans:
        df = pu.filter_away_rows_with_nan_in_columns(df,columns)
    return df[columns].to_numpy().astype('float')
    
    
def flatten_column_multi_index(df):
    df.columns = ["_".join(k) for k in df.columns.to_flat_index()]
    return df

def flatten_row_multi_index(df):
    return df.reset_index()

import numpy as np
def restrict_df_to_coordinates_within_radius_old(
    df,
    name,
    center,
    radius,
    within_radius = True,
    suffix = "nm",
    axes = ('x','y','z'),
    verbose = False,
    ):
    """
    Purpose: Restrict df by radius and a center point
    
    Ex: 
    import pandas_utils as pu

    center_df = hdju.seg_split_centroid(
        table = hdju.proofreading_neurons_with_gnn_cell_type_fine,
        features = [
            "gnn_cell_type_coarse",
            "gnn_cell_type_fine",
            "cell_type"
        ],
        return_df = True,
    )

    pu.restrict_df_to_coordinates_within_radius(
        center_df,
        name = "centroid",
        center = [861471,976721,896473],
        radius = 100_000
    )
    """
    center = np.array(center)
    coords = pu.coordinates_from_df(
        df,
        name=name,
        suffix=suffix,
        axes = axes,
    )
    
    dists = np.linalg.norm(coords-center,axis=1)
    if within_radius:
        idx_map = dists <= radius
        descr = "inside"
    else:
        idx_map = dists >= radius
        descr = "outside"
        
    if verbose:
        print(f"# of records {descr} radius ({radius}): {np.sum(idx_map)}")
        
    return df.iloc[idx_map,:].reset_index(drop=True)

# ------------- datajoint and pandas interface -------
import regex_utils as reu

def table_type_from_table(table):
    if pu.is_dataframe(table):
        return "pandas"
    elif "datajoint" in str(type(table)):
        return "dj"
    else:
        raise Exception("")



def query_str_from_list(
    restrictions,
    table_type="dj",
    verbose = False,
    joiner = "AND"):
    """
    Purpose: To turn any list of restrictions into a restriction str
    """
    restrictions= nu.to_list(restrictions)
    if len(restrictions) == 0:
        return None

    if table_type == "dj":
        restrictions = [k.replace("==","=") for k in restrictions]
        joiner = joiner.upper()

    else:
        restrictions = [k.replace(" = "," == ") for k in restrictions ]
        joiner = joiner.lower()

    restr_str = joiner.join([f"({k})" for k in restrictions])

    if verbose:
        print(f"query_str = {restr_str}")
    return restr_str

def query_str(restrictions,
    table_type="dj",
    verbose = False,
    joiner = "AND"):
    
    restrictions = nu.to_list(restrictions)
    return query_str_from_list(
        restrictions,
        table_type=table_type,
        verbose = verbose,
        joiner = joiner,
    )

restriction_str_from_list = query_str_from_list

dj_to_pandas_query_map = {" = ":" == ","AND":"and","OR":"or","NOT":"not","SQRT":"sqrt"}
def pandas_query_str_from_query_str(query):
    """
    Purpose: To replace a query of another form (usually datajoint query)
    to a pandas query
    
    Pseudocode:
    1) For all of the joiners (and, or, not), convert to lowercase
    2) convert the ' = ' to double equal
    """
    return reu.multiple_replace(
        query,
        dj_to_pandas_query_map
    )

def datajoint_query_str_from_query_str(query):
    return reu.multiple_replace(
        query,
        {v:k for k,v in dj_to_pandas_query_map.items()}
    )

def query_str_from_type(
    query,
    table_type = "pandas"
    ):
    
    if table_type == "pandas":
        return pu.pandas_query_str_from_query_str(query)
    elif table_type == 'dj':
        return pu.datajoint_query_str_from_query_str(query)
    else:
        raise Exception("")
        
def query_table_from_query_str(
    table,
    query):
    
    if pu.is_dataframe(table):
        query = pu.query_str_from_type(query,table_type = 'pandas')
        return table.query(query)
    else:
        query = pu.query_str_from_type(query,table_type = 'dj')
        return table & query
    
restrict_df_from_str = query_table_from_query_str
    
def query_table_from_list(
    table,
    restrictions,
    verbose = False,
    verbose_filtering = False,
    joiner = "AND",
    return_idx = False
    ):
    
    
    if len(restrictions) == 0:
        return table

    table_type = pu.table_type_from_table(table)
    query = pu.query_str_from_list(
        restrictions,
        table_type=table_type,
        verbose = verbose,
        joiner=joiner,)
    st = time.time()
    return_df =  pu.query_table_from_query_str(table,query)
    if verbose_filtering:
        print(f"Reduced table to {len(return_df)}/{len(table)} entries ")
    #print(f"Time for query = {time.time() - st}")
    if return_idx:
        return return_df.index.to_numpy()
    return return_df

restrict_df_from_list = query_table_from_list

def bbox_query(
    coordinate_name="centroid",
    
    # -- arguments for creating the bounding box
    center = None,
    buffer = None,
    bbox = None,
    buffer_array=None,
    buffer_percentage = None,
    buffer_array_multipier = None,
    
    # ---- for creating query
    suffix = None,
    inside = True,
    upper_case = False,
    axes = ('x','y','z'),
    
    verbose = False,
    ):
    """
    Purpose: Want to create a query that
    will restrict coordinates to a rectangular
    window

    """

    
    bbox = nu.bbox_with_buffer(
        center = center,
        buffer = buffer,
        bbox = bbox,
        buffer_array=buffer_array,
        percentage = buffer_percentage,
        multiplier = buffer_array_multipier,
        subtract_buffer = False,
    )


    if suffix is None:
        suffix = ""

    if upper_case:
        joiner = "AND"
        notter = "NOT"
    else:
        joiner = "and"
        notter = "not"


    inequalities = []
    for ineq,array in zip([">=","<="],bbox):
        inequalities += [f"({coordinate_name}_{xyz}{suffix} {ineq} {array[i]})" for i,xyz in enumerate(["x","y","z"])]

    if verbose:
        print(f"inequalities = {inequalities}")

    query = f" {joiner} ".join(inequalities)

    if not inside:
        query = f"({notter} ({query}))"

    if verbose:
        print(f"bbox query = {query}") 
        
    return query


def radius_query(
    center,
    radius,
    coordinate_name = "centroid",
    suffix = "",
    axes = ("x","y","z"),
    verbose = False,
    inside = True,
    upper_case = False,
    ):
    """
    Purpose: Query to Find if coordintes are inside a radius

    Pseudocode: 
    Design a query that computes
    sqrt((x - x_coord)**2 + (y - y_coord)**2 + (z - z_coord)**2) < radius
    """
    if upper_case:
        sqrt_func = "SQRT"
    else:
        sqrt_func = "sqrt"
    
    query = "+".join([f"(({coordinate_name}_{ax}{suffix} - {coord}) * ({coordinate_name}_{ax}{suffix} - {coord}))" for ax,coord in zip(axes,center)])

    if inside:
        comp = "<="
    else:
        comp = ">="
    query = f"{sqrt_func}({query}) {comp} {radius}"

    if verbose:
        print(f"radius query = {query}")

    return query

def restrict_df_to_coordinates_within_radius(
    df,
    name,
    center,
    radius,
    within_radius = True,
    suffix = "_nm",
    axes = ('x','y','z'),
    verbose = False,
    ):
    
    curr_query = pu.radius_query(
        center = center,
        radius = radius,
        inside = within_radius,
        suffix = suffix,
        axes = axes,
        verbose = verbose,
        coordinate_name = name,
    )
    
    return df.query(curr_query)

def distance_between_coordinates(
    df,
    coordinate_column_1,
    coordinate_column_2,
    coordinate_column_1_suffix="nm",
    coordinate_column_2_suffix="nm",
    axes = ('x','y','z')
    ):

    axes = list(axes)
    return np.sqrt(
        np.sum(np.array([(df[f"{coordinate_column_1}_{ax}_{coordinate_column_1_suffix}"]-df[f"{coordinate_column_2}_{ax}_{coordinate_column_2_suffix}"])**2
               for ax in axes]).T.astype('float'),axis = 1)
    )

def vector_between_coordinates(
    df,
    coordinate_column_1,
    coordinate_column_2,
    coordinate_column_1_suffix="nm",
    coordinate_column_2_suffix="nm",
    ):

    return (np.array([(df[f"{coordinate_column_1}_{ax}_{coordinate_column_1_suffix}"]-df[f"{coordinate_column_2}_{ax}_{coordinate_column_2_suffix}"])
               for ax in ["x","y","z"]]).T.astype('float')
    )

def distance_from_vector(
    df,
    vector_column,
    vector_column_suffix="nm",
    ):

    return np.sqrt(
        np.sum(np.array([(df[f"{vector_column}_{ax}_{vector_column_suffix}"])**2
               for ax in ["x","y","z"]]).T.astype('float'),axis = 1)
    )



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

pandas_alternatives_for_large_datasets = dict(
    ray = None,
    dask = None,
    modin = "https://modin.readthedocs.io/en/stable/"
)


def source_target_coordinate_edges(
    df,
    source_column = "presyn_centroid",
    target_column = "postsyn_centroid",
    suffix='nm',
    **kwargs):
    """
    Purpose: Get edges from source, target
    coordinate edges 
    """
   
    pre_coords = pu.coordinates_from_df(df,source_column,suffix=suffix,**kwargs)
    post_coords = pu.coordinates_from_df(df,target_column,suffix=suffix,**kwargs)
    pre_post_edges = np.hstack([pre_coords,post_coords]).reshape(-1,2,3)
    return pre_post_edges


def split_df_to_source_target_df(
    df,
    columns = None,
    source_name = "presyn",
    target_name = "postsyn",
    append_type = "prefix",
    source_columns = None,
    target_columns = None,
    include_source_target_name_in_columns = True,
    source_target_name_not_in_eachother_columns = True,
    verbose = False,
    ):
    """
    Purpose: To split a dataframe into source and 
    target dataframe

    Pseudocode: 
    1) Divide the dataframe into the source column, target columns
    """

    def column_to_name(column,append_name):
        if len(column) == 0:
            return append_name
        if append_type == "prefix":
            return f"{append_name}_{column}"
        elif append_type == "suffix":
            return f"{column}_{append_name}"
        elif append_type is None:
            return f"{column}"
        else:
            raise Exception("")

    def append_name_in_columns(column,append_name):
        if append_type == "prefix":
            return append_name == column[:len(append_name)]
        elif append_type == "suffix":
            return append_name == column[-len(append_name):]
        else:
            raise Exception("")

    if source_columns is None:
        if columns is not None:
            source_columns = [
                column_to_name(k,source_name) for k in columns
            ]
        else:
            source_columns = [k for k in df.columns if append_name_in_columns(k,source_name)]

        if source_target_name_not_in_eachother_columns:
            source_columns = [k for k in source_columns if target_name not in k]

    if include_source_target_name_in_columns:
        if source_name not in source_columns:
            source_columns = [source_name] + source_columns
            
    source_columns = [k for k in source_columns if k in df.columns]
            
    if verbose:
        print(f"\nsource_columns ({len(source_columns)}) = {source_columns}")

    if target_columns is None:
        if columns is not None:
            target_columns = [
                column_to_name(k,target_name) for k in columns
            ]
            
        else:
            target_columns = [k for k in df.columns if append_name_in_columns(k,target_name)]

        if source_target_name_not_in_eachother_columns:
            target_columns = [k for k in target_columns if source_name not in k]

    if include_source_target_name_in_columns:
        if target_name not in target_columns:
            target_columns = [target_name] + target_columns
            
    target_columns = [k for k in target_columns if k in df.columns]
            
    if verbose:
        print(f"\ntarget_columns ({len(target_columns)}) = {target_columns}")

    return df[source_columns],df[target_columns]

def randomly_sample_source_target_df(
    df=None,
    
    n_samples = 100,
    replace = True,
    seed=None,
    
    # -- arguments for the source_target_split
    source_df = None,
    target_df = None,
    
    columns = None,
    source_name = "presyn",
    target_name = "postsyn",
    append_type = "prefix",
    source_columns = None,
    target_columns = None,
    include_source_target_name_in_columns = False,
    source_target_name_not_in_eachother_columns = True,
    ):
    """
    Purpose: To create a random sampling of source
    and target from a source-target dataframe. This is to 
    create synthetic data
    """

    if source_df is None or target_df is None:
        source_df,target_df = pu.split_df_to_source_target_df(
            df = df,
            columns = columns,
            source_name = source_name,
            target_name = target_name,
            append_type = append_type,
            source_columns = source_columns,
            target_columns = target_columns,
            include_source_target_name_in_columns = include_source_target_name_in_columns,
            source_target_name_not_in_eachother_columns = source_target_name_not_in_eachother_columns,
        )

    source_df_samp = pu.randomly_sample_df(
        source_df,
        n_samples = n_samples,
        replace=replace,
        seed = seed,
    )

    target_df_samp = pu.randomly_sample_df(
        target_df,
        n_samples = n_samples,
        replace=replace,
        seed = seed,
    )

    df_samp = pu.concat([source_df_samp,target_df_samp],axis = 1)
    return df_samp
    
def replace_str_characters(
    df,
    replace_dict = None,
    column = None,
    to_replace = None,
    value = None,
    in_place = False,
    ):
    """
    Purpose: want to replace a string
    character with another in a column/columns
    of a dataframe
    """

    if not in_place:
        df = df.copy()
    
    if column is None:
        column = list(df.columns)
    
    column = list(nu.array_like(column))
    if replace_dict is None:
        replace_dict = dict(to_replace=value)

    for c in column:
        for k,v in replace_dict.items():
            df[c] = df[c].str.replace(k,v)

    return df

import numpy_utils as nu
def keys_from_groupby_obj(obj):
    """
    Purpose: get the groupby object
    keys
    """
    return obj.groups.keys()

def split_df_by_groupby_column(
    df,
    column,
    return_keys = False,
    verbose = False,
    ):
    """
    Purpose: Want to divide a dataset by a groupby statement
    """
    st = time.time()
    column = list(nu.array_like(column))
    gb = df.groupby(column)    
    all_dfs = [gb.get_group(x) for x in gb.groups]
    
    if verbose:
        print(f"# of groups = {len(all_dfs)} (total time = {time.time() - st})")
        
    if return_keys:
        keys = gb.groups.keys()
        return all_dfs,keys
    else:
        return all_dfs
    
    
def append_to_column_names(
    df,
    append_str,
    append_type = "prefix",
    columns = None,
    columns_to_omit = None,
    verbose = False,
    ):
    if append_type == "prefix":
        curr_str = 'f"{append_str}_{k}"'
    elif append_type == "suffix":
        curr_str = 'f"{k}_{append_str}"' 
    else:
        raise Exception("")
        
    if columns is None:
        columns = list(df.columns)
    if columns_to_omit is None:
        columns_to_omit = []
    
    append_dict = {k:eval(curr_str,locals(),globals()) 
                  for k,append_str in zip(columns,[append_str]*len(columns))
                  if k not in columns_to_omit }
    
    if verbose:
        print(f"append_dict= {append_dict}")
    
    return pu.rename_columns(
        df,
        append_dict
    )


def merge_dfs_with_same_columns(
    df,
    df_append,
    on,
    how = "inner",
    append_str="2",
    append_type='suffix',
    ):
    """
    Purpose: to merge 2 dataframes
    with same columns but want to keep them distinct
    """
    on = nu.array_like(on)
    
    df_append = pu.append_to_column_names(
        df_append,
        append_str=append_str,
        append_type=append_type,
        columns_to_omit = on ,
        
    )
    
    df_after_merge = pd.merge(
        df,
        df_append,
        on = on,
        how = how,
    )
    
    return df_after_merge

def dict_column_to_columns(
    df,
    column,
    verbose = False,
    return_new_columns = False,
    ):
    """
    Purpose: to unpack a column that is a list of dicts
    into columns themselves
    """
    
    df_dicts = pd.DataFrame.from_records(df[column].to_list())
    
    if verbose:
        print(f"New columns = {df_dicts.columns}")
    df_with_dicts = pu.concat([
        pu.delete_columns(df,column),df_dicts
    ],axis=1)
    
    if return_new_columns:
        return df_with_dicts,list(df_dicts.columns)
    else:
        return df_with_dicts
    
def map_column_with_dict_slow(
    df,
    column,
    dict_map,
    default_value=None,
    verbose = False,
    in_place = False,
    **kwargs
    ):
    if not in_place:
        df = df.copy()
    if verbose:
        st = time.time()
    
    df[column] = pu.new_column_from_dict_mapping(
        df,
        dict_map=dict_map,
        column_name=column,
        default_value=default_value,
    )
    if verbose:
        print(f"Total time = {time.time() - st}")
    
    return df




def map_column_with_dict(
    df,
    column,
    dict_map,
    use_default_value = True,
    default_value = None,
    in_place = False,
    verbose = False,
    ):
    
    
    
    if verbose:
        st = time.time()
        
    if not in_place:
        df = df.copy()
        
    columns = nu.to_list(column)
    
    for column in columns:
        original_labels = df[column].to_numpy().copy()
        for k,v in dict_map.items():
            if type(k) == str:
                query = (f"{column} == '{k}'")
            else:
                print(f"int query")
                query = (f"{column} == {k}")
            df = pu.set_column_subset_value_by_query(
                df,
                query=query,
                column = column,
                value = v

            )

        if use_default_value:
            # # --- setting the default value ---
            mask = np.invert(nu.mask_of_array_1_elements_in_array_2(
                original_labels,
                list(dict_map.keys())
            ))  

            df.loc[mask,column] = default_value
        
    if verbose:
        print(f"Total time = {time.time() - st}")
    return df

def set_categorical_order_on_column(
    df,
    column,
    order,
    order_in_least_to_greatest = False,
    ):
    
    if not order_in_least_to_greatest:
        order = list(np.flip(order))
    
    df[column] = df[column].astype("category")
    df[column] = df[column].cat.set_categories(order)
    return df

def sort_df_by_categorical_column(
    df,
    column,
    order,
    order_in_least_to_greatest = False,
    ascending = False,
    ):
    
    """
    Purpose: To order a dataframe from 
    a categorical column
    """
    
    df_with_order = set_categorical_order_on_column(
        df,
        column,
        order = order,
        order_in_least_to_greatest=order_in_least_to_greatest,
    )

    return pu.sort_df_by_column(
        df_with_order,
        columns=column,
        ascending = ascending)
    
df_of_coordinates_and_labels_from_dict = nu.df_of_coordinates_and_labels_from_dict    

def aggregate_over_column(
    df,
    group_column,
    column,
    aggregator = "sum",
    divisor = None
    ):
    """
    Purpose: to aggregate a column over
    the dataframe
    
    Ex: 
    pu.aggregate_over_column(
        edits_df,
        group_column = "split_type",
        column = "error_branches_skeleton_length",
        divisor = 1000,
    )
    """

#     aggregator = "sum"
#     column = "error_branches_skeleton_length"
#     group_column = "split_type"
#     divisor = 1000
    aggr_df = getattr(df[[group_column,column]].groupby([group_column]),aggregator)()
    if divisor is not None:
        aggr_df[column] = aggr_df[column]/divisor
    return aggr_df


def label_to_coordinate_dict_from_df(
    df,
    label_column = "label",
    coordinate_prefix = None,
    coordinate_suffix = None,):
    """
    Purpose: To convert a dataframe with coordinatea and labels
    into a dictionary mapping the labels to the coordinates
    """

    df_divided, labels = pu.divide_dataframe_by_column_value(
        df,
        label_column
    )
    coord_dict = dict([(lab,pu.coordinates_from_df(
        k,
        name = coordinate_prefix,
        suffix = coordinate_suffix))
     for k,lab in zip(df_divided,labels)])

    return coord_dict


def xlim_ylim_from_coordinate_df(
    df,
    x = "x",
    y = "y"
    ):
    
    xlim = [df[x].min(),df[x].max()]
    ylim = [df[y].min(),df[y].max()]
    
    return xlim,ylim

def order_columns(
    df,
    columns_at_front=None,
    columns_at_back=None,
    ):
    """
    Purpose: To order the columns of  
    a dataframe
    
    Pseudocode: 
    
    """
    if columns_at_front is None:
        columns_at_front = []
    if columns_at_back is None:
        columns_at_back= []
        
    columns_at_front = list(columns_at_front)
    columns_at_back = list(columns_at_back)
    
    columns = list(df.columns)
    columns_left_over = np.setdiff1d(columns,columns_at_front)
    columns_left_over = np.setdiff1d(columns_left_over,columns_at_back)

    return df[columns_at_front + list(columns_left_over) + columns_at_back]
                  
def normalize_to_sum_1(
    df,
    axis = 1):
    
    return df.div(df.sum(axis=axis), axis=1-axis)

def restrict_df_by_min_max(
    df,
    verbose = False,
    plot = False,
    columns_to_plot = None,
    **kwargs):
    """
    Purpose: To restrict a dataframe
    by min and max of columns and just
    tell which columns and what min through 
    arguments
    
    Ex: 
    pu.restrict_df_by_min_max(
        spine_df_trans_umap,
        verbose = True,
        umap_1_max = -2.5,
        umap_0_max = 5,
        umap_0_min = 0
    )
    """
    restrictions = []
    columns_used = []
    for k,v in kwargs.items():
        col = k[:-4]
        if k[-4:] == "_min":
            restrictions.append(f"{col} >= {v}")
        elif k[-4:] == "_max":
            restrictions.append(f"{col} <= {v}")
        else:
            raise Exception("")
            
        if col not in columns_used:
            columns_used.append(col)

    if verbose:
        print(f"restrictions = {restrictions}")
    df_restr = pu.query_table_from_list(
        df,
        restrictions,
        verbose_filtering=verbose
    ).reset_index(drop=True)
    
    if plot:
        if columns_to_plot is None:
            columns_to_plot = columns_used
        area_coords = df_restr[columns_to_plot].to_numpy()
        whole_coords = df[columns_to_plot].to_numpy()
        fig,ax = plt.subplots(1,1,)
        ax.scatter(whole_coords[:,0],whole_coords[:,1],label = "unrestricted")
        ax.scatter(area_coords[:,0],area_coords[:,1],label = "restricted")
        ax.legend()
        ax.set_xlabel(columns_to_plot[0])
        ax.set_ylabel(columns_to_plot[1])
        plt.show()
    
    return df_restr

def closest_row_to_coordinate(
    df,
    coordinate,
    coordinate_name = "mesh_center",
    return_df = False,
    verbose = False,
    **kwargs
    ):
    """
    Purpose: To find the row idx of the 
    rows that have the closest coordinate to 
    a coordinate

    Pseudocode: 
    1) Export the coordinates of dataframe
    2) Find the closest idx to each coordinate
    3) return the idx or a dataframe
    
    Ex: 
    pu.closest_row_to_coordinate(
        df = spine_df,
        coordinate_name = "mesh_center",
        coordinate = np.array([829181.89697582, 494363.38581494, 899399.53232649]),
        return_df = True,
    )
    """



    
    if coordinate.ndim == 1:
        single_flag = True
    else:
        single_flag = False

    coords = pu.coordinates_from_df(
        df,
        name=coordinate_name,
        **kwargs
    )

    idx = nu.closest_idx_for_each_coordinate(coordinate,array_for_idx=coords)

    if verbose:
        print(f"Closest rows = {idx}")

    if return_df:
        return df.iloc[idx,:]
    else:
        if single_flag:
            return idx[0]
        return idx
    
def randomly_sample_classes_from_df(
    df,
    column,
    n_samples=None,
    classes = None,
    seed = None,
    verbose = False,):
    """
    Purpose: To randomly sample from each
    class from a column and concatenate the results
    """
    if n_samples is None:
        n_samples = len(df)
        
    if type(n_samples) == dict:
        samples_dict = n_samples
    else:
        samples_dict = dict()

    df = pu.shuffle_df(df,seed=seed)
    if classes is None:
        classes = df[column].unique()

    all_dfs = []
    for ct in classes:
        curr_df = df.query(f"{column} == '{ct}'"
            ).reset_index(drop=True).iloc[:samples_dict.get(ct,n_samples),:]
        if verbose:
            print(f"{ct}: # of samples = {len(curr_df)}")
        all_dfs.append(curr_df)

    all_dfs = pu.concat(all_dfs,axis = 0).reset_index(drop=True)
    return all_dfs

def min_max_from_column(
    df,
    column = None,
    return_dict = False,
    buffer_perc = 0):
    if column is None:
        column= df.columns
        
    if not nu.is_array_like(column):
        column = [column]
        single_flag = True
    else:
        single_flag = False
        
    min_maxes = dict()
    for c in column:
        min_v,max_v = df[c].min(),df[c].max()
        if buffer_perc is not None and buffer_perc > 0:
            buffer_v = (max_v - min_v)*(buffer_perc/100)
            min_v = min_v - buffer_v
            max_v = max_v + buffer_v
        min_maxes[c] = (min_v,max_v)
        
    min_maxes_values = list(min_maxes.values())
    
    if single_flag:
        return min_maxes_values[0]
    else:
        if return_dict:
            return min_maxes
        else:
            return min_maxes_values
        
        
def weighted_average_df(
    df,
    group_by_columns,
    weight_column,
    columns_to_delete = None,
    verbose = False,
    flatten = True,
    ):
    """
    Purpose: Compose a weighted average
    dataframe from a column in a dataframe 
    serving as the weights and 
    all the rest of the columns being averaged
    
    Ex: 
    curr_df = df.query(f"compartment == 'basal'").reset_index(drop=True)
    print(len(curr_df))
    curr_df_w = pu.weighted_average_df(
        curr_df,
        weight_column = "width",
        columns_to_delete = ["node",],
        group_by_columns = ["segment_id","split_index","compartment"],
        verbose = True,
    )
    """
    if verbose:
        st = time.time()
        
    df_to_group = df
    df_to_group = df_to_group.query(f"{weight_column} > 0").reset_index(drop=True)
    if columns_to_delete is not None:
        df_to_group = pu.delete_columns(df_to_group,columns_to_delete)

    wm = lambda x: np.average(x,weights = df_to_group.loc[x.index,weight_column])
    df_weight = df_to_group.groupby(group_by_columns).agg(
        wm
    )
    
    if verbose:
        print(f"Total time for weighted average = {time.time() - st}")
        
    if flatten:
        df_weight = pu.flatten_row_multi_index(df_weight)

    return df_weight
    
def normalize_vector_magnitude(
    df,
    name="synapse",
    suffix = "nm",
    axes = ("x","y","z"),
    columns = None,
    in_place = False,
    ):

    """
    Purpose: Want to vector noramlize
    a group of columns in a dataframe

    Psuedocode: 
    1) Construct the column names that will be extracted
    2) extract the coordinates
    3) Compute the norm
    4) Divide column by the norm
    """
    
    if not in_place:
        df = df.copy()

    columns = pu.coordinate_columns(
        name=name,
        suffix = suffix,
        axes = axes,
        )

    coordinates = pu.coordinates_from_df(
        df,
        columns = columns,
    )

    magn = np.linalg.norm(coordinates,axis = 1).reshape(-1,1)
    df[columns] = coordinates/magn
    return df

def reorder_levels(df,order,axis = 0):
    """
    Purpose: Will reorder the columns or row indexes
    if multi level
    """
    return df.reorder_levels(order,axis = axis)

def unstack(df,level = -1):
    """
    Can cnvert a index to a column index
    """
    return df.unstack(level=level)

def convert_series_to_df(s):
    return s.to_frame()

def unravel_df_into_single_row_with_col_prefixes(
    df,
    columns,
    ):
    """
    Purpose: If did a groupby and have columns that represent
    unique combinations then can make dataframe one row dataframe
    and put those column combinations as prefixes to columns
    """
    columns = nu.to_list(columns)
    new_order = list(range(1,len(columns)+1)) + [0]
    v = df.set_index(columns).unstack(columns).to_frame().reorder_levels(order = new_order).T
    v.columns = v.columns.map('_'.join)
    return v




def group_df_for_count_and_average(
    df,
    columns,
    default_value = 0,
    return_one_row_df = False,):
    """
    Purpose: To group columns by averaging and counting
    """
    columns = nu.to_list(columns)
    df_lite = df
    
    features = [k for k in df_lite.columns if k not in columns]
    df_lite = pu.replace_nan_with_default(df_lite,default=default_value)
    df_lite = pu.replace_None_with_default(df_lite,default=default_value)
    df_lite[features] = df_lite[features].astype('float')

#     reduced_df = pu.flatten_row_multi_index(df_lite.groupby(
#         columns
#     ).mean())

    reduced_df = df_lite.groupby(
        columns
    ).mean()  

    cont_df = pu.count_unique_column_values(df_lite,columns)

    stats_df = pd.merge(
        reduced_df,
        cont_df,
        how = "left",
        on = columns,
    )
    
    if return_one_row_df:
        stats_df = unravel_df_into_single_row_with_col_prefixes(
            stats_df,
            columns
        )
    
    return stats_df

def divide_df_by_column_bins(
    df,
    column,
    bins = None,#equal_width,#equal_depth
    n_bins = 10,
    #arguments for filtering
    percentile_buffer=None,
    percentile_lower=None,
    percentile_upper=None,

    overlap = 0,
    verbose = True,
    return_bins = False,
    flatten_return_bins = False,
    ):
    """
    Purpose: Want to split up a dataframe
    with bins or number of bins perscribed
    over a column value
    """
    intervals = bins
    n_intervals = n_bins
    interval_attribute = column

    if type(intervals) == str:
        intervals = nu.bin_array(
            df[interval_attribute].to_numpy(),
            n_bins = n_intervals,
            bin_type = intervals
        )
        intervals = np.vstack([intervals[:-1],intervals[1:]]).T
    elif intervals is not None:
        intervals = np.array(intervals)
        if intervals.ndim <= 1:
            intervals = np.vstack([intervals[:-1],intervals[1:]]).T

    df = df.query(f"{interval_attribute} == {interval_attribute}")

    df = pu.filter_df_by_column_percentile(
        df,
        columns=interval_attribute,
        percentile_buffer=percentile_buffer,
        percentile_lower = percentile_lower,
        percentile_upper = percentile_upper,
        verbose = verbose,
        )

    #2) Get the continuous value you will bin and divide it up intervals
    interval_vals = df[interval_attribute]
    if intervals is None:
        intervals = nu.interval_bins_covering_array(
            array = interval_vals,
            n_intervals = n_intervals,
            overlap = overlap, #if this is a percentage then it is a proportion of the interval
            outlier_buffer = 0,
            verbose = False,
        )

    all_dfs = []
    for j,(lower,upper) in enumerate(intervals):
        df_curr = df.query(
            f"({interval_attribute} >= {lower})"
            f"and ({interval_attribute} <= {upper})"
        )

        all_dfs.append(df_curr)
        if verbose:
            print(f"[{lower:.2f},{upper:.2f}]: {len(df_curr)} samples ")


    if return_bins:
        if flatten_return_bins:
            intervals = np.hstack([intervals[:,0],[intervals[-1,-1]]])
        return [all_dfs,intervals]
    else:
        return all_dfs

    
import seaborn as sns
def histogram_2d(
    df,
    x,
    y,
    n_x_bins = 10,
    n_y_bins = 10,
    x_bins = None,
    y_bins = None,
    verbose = False,
    return_bins = True,
    normalize_rows = True,
    plot = False,
    return_df = False,
    **kwargs
    ):
    """
    Purpose: compute a 2D
    histogram array of 2 columns
    in a dataframe
    """
    if x_bins is None:
        x_bins = nu.equal_width_bins(
            df[x].to_numpy(),
            n_x_bins
        )

    ret_dfs,y_bins = pu.divide_df_by_column_bins(
        df = df,
        column = y,
        bins=y_bins,
        n_bins=n_y_bins,
        verbose = verbose,
        return_bins=True,
        flatten_return_bins = True,
        **kwargs
    )

    hists = []
    for curr_df in ret_dfs:
        hist_values,_ = np.histogram(
            curr_df[x].to_numpy(),
            bins = x_bins,
        )

        hists.append(hist_values)

    hists = np.vstack(hists)

    if normalize_rows:
        hists = (hists/(hists.sum(axis=1).reshape(-1,1)))
        
    stat_df = pd.DataFrame(hists)
    stat_df.columns = np.vstack([x_bins[1:],x_bins[:-1]]).mean(axis = 0)
    stat_df.index = np.vstack([y_bins[1:],y_bins[:-1]]).mean(axis = 0)
    if plot:
        sns.heatmap(
            data = stat_df,
            cmap = "Blues"
        )
    
    if return_df:
        hists = stat_df
    if return_bins:
        return hists,x_bins,y_bins
    else:
        return hists
    
    
def new_column_from_str_func(
    df,
    funcs):

    if type(funcs)== dict:
        funcs = [f"{k}={v}"for k,v in funcs.items()]
    if type(funcs) == str:
        funcs = [funcs]
        
    curr_df = df
    for func in funcs:
        curr_df = curr_df.eval(func)
        
    return curr_df

def new_column_from_name_to_str_func_dict(
    df,
    func_dict):
    
    return new_column_from_str_func(
    df,
    func_dict)

import matplotlib_utils as mu
import numpy_utils as nu
import ipyvolume_utils as ipvu
def plot_class_coordinates(
    df,
    column,
    classes = None,
    centroid_column = "centroid",
    suffix = "nm",
    classes_colors = None,
    verbose = True,
    size = 0.5,
    **kwargs
    ):
    """
    Purpose: to plot the excitatory and inhibitory cells
    in a dataframe that has their coordinates
    """

    if classes is None:
        classes = df[column].unique()

    if classes_colors is None:
        classes_colors = mu.generate_non_randon_named_color_list(len(classes))
    elif type(classes_colors) == dict:
        classes_colors = [classes_colors[k] for k in classes]
    else:
        classes_colors = nu.to_list(classes_colors)

    scatters = []
    scatters_colors = []

    for cl,col in zip(classes,classes_colors):
        coords = pu.coordinates_from_df(
                df.query(f"{column} == '{cl}'"),
                name = centroid_column,
                suffix=suffix,
        )
        if verbose:
            print(f"# of coords for {cl} ({col}) = {len(coords)}")

        scatters.append(
            coords
        )

        scatters_colors.append(col)
    
    
    ipvu.plot_multi_scatters(
        scatters=scatters,
        color=scatters_colors,
        size = size,
        **kwargs
        

    )
    
def bin_idx_for_column(
    df,
    column,
    n_bins = 20,
    add_bin_idx = True,
    bin_idx_name = None,
    add_bin_midpoint = True,
    bin_midpoint_name = None,
    in_place = False,
    verbose = False,
    return_df = True,
    bin_idx_name_default = "bin_idx",
    bin_midpoint_name_default = "bin_midpoint"
    ):
    """
    Purpose: to assign a bin value for a column
    in a dataframe
    """

    if return_df and (not add_bin_idx and not add_bin_midpoint):
        raise Exception("Doing nothing")


    if not in_place and (add_bin_idx or add_bin_midpoint):
        df = df.copy()

    column_data = df[column].to_numpy()
    bins = nu.bin_array(
        column_data,
        n_bins = n_bins,
    )

    bin_assignment = np.digitize(
        column_data,
        bins = bins
    ) - 1

    bin_assignment[bin_assignment >= n_bins ] = n_bins-1
    bin_mids = (bins[1:] + bins[:-1])/2

    if add_bin_idx:
        if bin_idx_name is None:
            bin_idx_name= f"{column}_{bin_idx_name_default}"
        df[bin_idx_name] = bin_assignment

    if add_bin_midpoint:
        if bin_midpoint_name is None:
            bin_midpoint_name= f"{column}_{bin_midpoint_name_default}"
        df[bin_midpoint_name] = bin_mids[bin_assignment]

    if verbose:
        display(pu.count_unique_column_values(df,bin_idx_name))
    # if plot:
    #     dummy_name = f"{bin_midpoint_name} "
    #     df[dummy_name] = [
    #         str(np.round(k,2)) for k in df[bin_midpoint_name].to_list()]
    #     df["count"] = 1
    #     sns.barplot(
    #         data=df,
    #         y = dummy_name,
    #         x = column,
    #         color = "blue"
    #     )

    #     df = pu.delete_columns(df,dummy_name)

    if return_df:
        return df
    else:
        return bin_assignment,bin_mids
def col_to_datatype_dict(df):
    cols = list(df.columns)
    dtypes = [str(k) for k in df.dtypes.to_list()]
    return {col:dt for col,dt in zip(cols,dtypes)}

def columns_of_datatype(
    df,
    datatype="float",
    verbose = False,):
    col_dt_dict = pu.col_to_datatype_dict(df)
    cols = [col for col,dt in col_dt_dict.items() if datatype in dt]
    if verbose:
        print(f"Columns of {datatype} datatype: \n{cols}")
    return cols

def float_columns(df,verbose = False,):
    return columns_of_datatype(
        df,
        datatype="float",
        verbose=verbose,
    )

def round_float_cols(
    df,
    precision = 2,
    column_precision_dict = None,
    verbose = False,
    cols_to_ignore = None):
    """
    Purpose: To round all float columns
    to a specified precision
    """

    if cols_to_ignore is None:
        cols_to_ignore = []
    if column_precision_dict is None:
        column_precision_dict= {}

    float_cols = pu.float_columns(df,verbose=verbose,)
    for k in float_cols:
        if k not in column_precision_dict:
            if k in cols_to_ignore:
                continue
            column_precision_dict[k] = precision
            
    return df.round(column_precision_dict)
        
def flip_rows(df):
    return df.iloc[::-1]
    
def mode_aggr_groupby(
    df,
    groupby_columns,
    column,
    ):
    
    groupby_columns= list(nu.to_list(groupby_columns))
    #columns= list(nu.to_list(columns))
    
    return df.groupby(groupby_columns)[column].agg(lambda x: pd.Series.mode(x)[0]).to_frame().reset_index()
    
def filter_df_splits_by_column_percentile(
    df,
    column,
    split_columns,
    percentile_upper=99.5,
    percentile_lower = 0,
    verbose = False,
    ):
    """
    Purpose: To filter multiple dataframe splits
    to a certain percentage and then stack into dataframe
    """
    if verbose:
        print(f"Before filtering {split_columns} column to {[percentile_lower,percentile_upper]} perc = {len(df)}")
    df = pu.concat(
        [pu.filter_df_by_column_percentile(k,columns = column,percentile_lower=percentile_lower,percentile_upper=percentile_upper)
         for k in pu.split_df_by_columns(df,columns = split_columns)],
    axis = 0
    ).reset_index(drop=True)

    if verbose:
        print(f"AFTER filtering {split_columns} column to {[percentile_lower,percentile_upper]} perc = {len(df)}")

    return df

def cumulative_count_within_groupby(
    df,
    group_columns,
    base_idx = 0,
    ):
    """
    Purpose: To cumutatively count
    the rows in a groupby
    """
    return df.groupby(group_columns).cumcount() + base_idx

def flatten_pivot_df(
    df,
    index_name = "index"):
    """
    Purpose: To flatten the results of a pivot table

    Pseudocode: 
    1) Rename the 
    """

    curr_df = df

    try:
        curr_df = pu.flatten_column_multi_index(curr_df)
    except:
        pass
    cols = curr_df.columns
    rename_dict = {k:f"{cols.name}_{k}" for k in cols}
    rename_dict

    curr_df = pu.flatten_row_multi_index(curr_df)
    curr_df = pu.delete_columns(
        (pu.rename_columns(curr_df,rename_dict)),
        columns_to_delete=[cols.name]
    )
    curr_df.columns.names = [index_name]
    return curr_df

def top_k_extrema_attributes_as_columns_by_group(
    df,
    column,
    group_columns,
    extrema = "largest",
    k = 2,
    suffix = "_idx",
    ):
    """
    Purpose: To create colums that are the k
    extrema of a certain attribute within a group

    Pseudocode: 
    1) Filter df to k extrema of column inside group
    2) Label the idx of each row inside the group
    3) Create a pivot table to make those rows as columns
    4) Flatten pivot table
    """
    sort_column = column

    # Purpose: Want to collapse to the top k widths of a certain compartment
    df_sort_top_k = pu.filter_to_extrema_k_of_group(
        df,
        group_columns=group_columns,
        sort_columns=sort_column,
        k = k,
        extrema=extrema,

    )

    df_sort_top_k[f"{sort_column}{suffix}"] = pu.cumulative_count_within_groupby(
        df = df_sort_top_k,
        group_columns = group_columns,
    )

    df_sort_pivot = df_sort_top_k.pivot(
        index = group_columns,
        columns = f"{sort_column}{suffix}",
        values=sort_column
    )
    df_sort_pivot = pu.flatten_pivot_df(
        df_sort_pivot,
    )

    return df_sort_pivot

def bin_for_column(
    df,
    column,
    n_bins = 10,
    as_str = True,
    generate_color_palette = True,
    bin_name = "bin",
    divisor = 1,
    #as_str_precision = 2
    ):
    
    df_to_plot = pu.bin_idx_for_column(
        df = df,
        column = column,
        n_bins=n_bins,
    )
    
    bin_to_plot = bin_name
    
    df_to_plot[bin_to_plot] = [
        float(k)/divisor for k in df_to_plot[f"{column}_bin_midpoint"]
    ]

    df_to_plot = pu.sort_df_by_column(
        df_to_plot,
        columns=bin_to_plot,
        ascending=True
    )

    if as_str:
        df_to_plot[bin_to_plot] = [f"{k:.2f}" for k in df_to_plot[bin_to_plot]]
    
    if generate_color_palette:
        unique_bins = df_to_plot[bin_to_plot].unique()
        color_palette = {k:v for k,v in zip(
            unique_bins,mu.generate_non_randon_named_color_list(len(unique_bins))
        )}
        
        return df_to_plot,color_palette
    return df_to_plot
    
import matplotlib_utils as mu
plot_gradients_over_coordiante_columns = mu.plot_gradients_over_coordiante_columns
                  

import pandas_utils as pu
