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
    

def turn_off_scientific_notation(n_decimal_places=3):
    pd.set_option('display.float_format', lambda x: '%.0f' % x)
    
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

def rename_columns(df,columns_name_map):
    return df.rename(columns=columns_name_map)

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

def divide_dataframe_by_column_value(df,
                                column,
                                ):
    """
    Purpose: To divide up the dataframe into 
    multiple dataframes
    
    Ex: 
    divide_dataframe_by_column_value(non_error_cell,
                                column="cell_type_predicted")
    
    """
    tables = []
    table_names = []
    for b,x in df.groupby(column):
        table_names.append(b)
        tables.append(x)
        
    return tables,table_names


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

def df_to_render_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                       fontsize_header=None,
                     ax=None,transpose=True,**kwargs):
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
    if transpose:
        data = data.transpose()
    
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if not transpose:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    else:
        mpl_table = ax.table(cellText=data.values, bbox=bbox, rowLabels=data.index, **kwargs)

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
    if filename_str[:-4] != ".pkl":
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
def filter_away_rows_with_duplicated_col_value(
    df,
    col,
    ):
    
    col = nu.convert_to_array_like(col)
    return df[~df.duplicated(subset=col,keep=False)]

def filter_to_first_instance_of_unique_column(df,column_name):
    column_name = nu.convert_to_array_like(column_name) 
    return df.groupby(column_name).first()

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
    ):
    """
    Purpose: To normalize a pandas table with
    means and standard deviations of the columns
    """

    if column_means is None:
        column_means = df.mean(axis=0).to_numpy()

    if column_stds is None:
        column_stds = df.std(axis=0).to_numpy()

    normalized_df=(df-column_means)/column_stds
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
def filter_away_non_finite_rows(df):
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
    columns,
    percentile_buffer=None,
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
        att_vals = df[attribute].to_numpy()
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

def randomly_sample_df(
    df,
    n_samples=None,
    seed = None,
    replace = False,
    verbose = False,):
    
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

def shuffle_df(df,**kwargs):
    return pu.randomly_sample_df(
        df,
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
    value,
    verbose = False):
    """
    Purpose: Set column
    """
    
    curr_map = df.eval(query)
    if verbose:
        print(f"Number of rows in query = {curr_map.sum()}")
        
    df.loc[curr_map,column] = value
    return df

def count_unique_column_values(
    df,
    column,
    sort = False,
    count_column_name = "unique_counts"):
    column = list(nu.convert_to_array_like(column))
    return_df =  pu.unique_row_counts(df[column],count_column_name=count_column_name)
    if sort:
        return_df = pu.sort_df_by_column(return_df,count_column_name)
    return return_df

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
    ):
    """
    Purpose: To calculate sub dataframe
    after binning a certain column
    """

    

    if bins is None:
        if equal_depth_bins:
            bins = nu.equal_depth_bins(
                df[column].to_numpy()
            )
        else:
            bins = np.linspace(0,df[column].max(),n_bins + 1)

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
    stat_column=None,
    bins=None,
    equal_depth_bins = False,
    n_bins = 10,
    verbose = False,
    return_bins = True,
    return_df_len = True,
    return_std = True,
    func_std = None,
    plot = False,
    plot_n_data_points = True,
    
    
    ):
    """
    Purpose: to compute a statistic over
    binned sub dfs (where bins are determined by column)
    """

    df_bins,bins = pu.bin_df_by_column(
        df=df,
        column=column,
        bins=bins,
        n_bins = n_bins,
        verbose = verbose,
        return_bins = True,
        equal_depth_bins=equal_depth_bins,
        )
    
    if type(func) == str:
        df_stats = [k[func].mean() for k in df_bins]
        df_std = [k[func].std() for k in df_bins]
    else:
        df_stats = [func(k) for k in df_bins]
        df_std = [func_std(k) for k in df_bins]
    df_len = [len(k) for k in df_bins]
    
    
    if plot:
        twin_color = "blue"
        fontsize = 20

        figsize = (10,5)
        mid_bins = (bins[1:] + bins[:-1])/2

        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.plot(mid_bins,df_stats,)
        ax.set_xlabel(f"{column}")
        if type(func) == str:
            ax.set_ylabel(f"{func}")
        else:
            ax.set_ylabel(f"{func.__name__}")

        if plot_n_data_points and not equal_depth_bins:
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('# of Data Points',fontsize=fontsize,color=twin_color)  # we already handled the x-label with ax1
            ax2.tick_params(axis='y', labelcolor=twin_color)
            ax2.plot(mid_bins, df_len, color=twin_color)
        plt.show()

    if (not return_bins) and (not return_df_len) and (not return_std):
        return df_stats
    return_list = [df_stats]
    if return_bins:
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

def restriction_str_from_list(
    restrictions,
    verbose = False,
    joiner="and",
    ):
    
    if len(restrictions) == 0:
        return None
    
    restrictions = [k.replace(" = "," == ") for k in restrictions ]
    joiner = joiner.lower()
    
    restr_str = joiner.join([f"({k})" for k in restrictions])
        
    if verbose:
        print(f"restr_str = {restr_str}")
    return restr_str

def restrict_df_from_list(
    df,
    restrictions,
    verbose = False,
    joiner="and",
    ):
    
    return_df = df.query(pu.restriction_str_from_list(
       restrictions=restrictions,
        verbose = verbose,
        joiner=joiner, 
    ))
    
    if verbose:
        print(f"Before Restriction of df: # of entries = {len(df)}")
        print(f"After Restriction of df: # of entries = {len(return_df)}")
        
    return return_df

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
        df = df.copy(deep=True)

    for k in [source_name,target_name]:
        if append_type == "prefix":
            name_str = 'f"{kk}_{v}"'
        elif append_type == "suffix":
            name_str = 'f"{v}_{kk}"'
        elif append_type is None:
            name_str = 'f"{kk}"'
        else:
            raise Exception("")

        
        rename_dict = {v:eval(name_str) for kk,v in zip([k]*len(columns),columns)}
            
        
        
        if on is None:
            curr_on = [v for v in rename_dict.values() if v in df.columns]
        else:
            if len(np.intersect1d(nu.convert_to_array_like(on),list(rename_dict.values()))) == 0:
                curr_on = k
                rename_dict[on] = k
                
        
                
        df = pd.merge(
            df,
            pu.rename_columns(df_append[columns],
                              rename_dict),
            on=curr_on,
            how=how,
        )
    return df

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



def coordinates_from_df(
    df,
    name="synapse",
    suffix = "nm",
    axes = ("x","y","z"),
    ):
    if suffix is not None and len(suffix) > 0:
        suffix = f"_{suffix}"
    else:
        suffix = ""
    return df[
        [f"{name}_{a}{suffix}" for
        a in axes]].to_numpy().astype('float')
    
    
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
    ):

    return np.sqrt(
        np.sum(np.array([(df[f"{coordinate_column_1}_{ax}_{coordinate_column_1_suffix}"]-df[f"{coordinate_column_2}_{ax}_{coordinate_column_2_suffix}"])**2
               for ax in ["x","y","z"]]).T,axis = 1)
    )
    
    
    
import pandas_utils as pu
