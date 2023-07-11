
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from . import numpy_dep as np
import pandas as pd
import scipy
import sklearn
#from python_tools import numpy_utils as nu

def ensure_scalar(array):
    if nu.is_array_like(array):
        return len(array)
    else:
        return array

def precision(TP,FP):
    TP = ensure_scalar(TP)
    FP = ensure_scalar(FP)
    
    if (TP + FP) > 0:
        return TP/(TP + FP)
    else:
        return np.nan
    
def recall(TP,FN):
    TP = ensure_scalar(TP)
    FN = ensure_scalar(FN)
    
    if (TP + FN) > 0:
        return TP/(TP + FN)
    else:
        return np.nan
    
def f1(TP,FP,FN):
    
    
    curr_prec = precision(TP,FP)
    curr_recall = recall(TP,FN)
    
    if curr_prec + curr_recall > 0:
        return 2*(curr_prec*curr_recall)/(curr_prec + curr_recall)
    else:
        return np.nan
    
def calculate_scores(TP,FP,FN):
    return dict(precision=precision(TP,FP),
               recall=recall(TP,FN),
               f1=f1(TP,FP,FN))

#from python_tools import pandas_utils as pu
#import pandas as pd
def add_false_true_positive_negative_labels(
    df,
    y_true_label,
    y_pred_label,
    output_column_name="category",
    positive_value=True,
    negative_value=False):
    """
    Purpose: To add the TP,TN,FP,FN labels to a dataframe
    
    """
    def classification_category(row):
            classified = row[y_pred_label]
            truth = row[y_true_label]

            if classified == positive_value and truth == positive_value:
                return "TP"
            elif classified == negative_value and truth == negative_value:
                return "TN"
            elif classified == positive_value and truth == negative_value:
                return "FP"
            elif classified == negative_value and truth == positive_value:
                return "FN"
            else:
                raise Exception("")
    df = pd.DataFrame(df)
    df[output_column_name] = pu.new_column_from_row_function(df,
                                                            classification_category)
    
    return df

#from sklearn.metrics import confusion_matrix

def true_and_pred_labels_to_confusion_matrix(y_true,
                                             y_pred,
                                             labels=None,
                                             return_df = True,
                                                ):
    """
    Purpose: To turn a list of the 
    classifications into a confusion matrix
    
    Example:
    labels=["inhibitory","excitatory"]
    true_and_pred_labels_to_confusion_matrix(df_filtered["manual_label"],
                                             df_filtered["auto_label"],
                                            labels)
    """

    return confusion_matrix(y_true,y_pred,
                labels=labels)

def df_to_confusion_matrix(df,
                       y_true_label=None,
                      y_pred_label=None,
                      labels=None,
                          return_df=False):
    """
    Purpose: Dataframe with columns representing classes
    to the confusion matrix of the prediction
    
    Ex:
    stu.df_to_confusion_matrix(df_filtered,labels=["inhibitory","excitatory"])
    """
    if y_true_label is None and y_pred_label is None:
        y_true_label = df.columns[0]
        y_pred_label = df.columns[1]
    
    return stu.true_and_pred_labels_to_confusion_matrix(df[y_true_label],
                                         df[y_pred_label],
                                        labels)

#from sklearn.metrics import precision_recall_fscore_support

def true_and_pred_labels_to_precision_recall_f1score(
    y_true,
    y_pred,
    labels=None,
    positive_value=None,
    average=None,
    verbose = False,
    return_dict = False,
    binary = False,
    pos_label = None,
    ):
    """
    Arguments for average
    average:
    - micro : Calculate metrics globally by counting the total true positives, false negatives and false positives
    - macro : Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account
    
    
    
    """
    if labels is None:
        lables = np.unique(y_true)
        print(f"Using labels : {lables}")
        
    if binary:
        average = "binary"
        if pos_label is None:
            pos_label = labels[0]
    precision,recall,f1,_ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average=average,
        pos_label = pos_label)
    
    
    
    if positive_value is None or average is not None:
        pass 
    else:
        positive_idx = np.where(np.array(labels) == positive_value)[0][0]
        precision,recall,f1 = precision[positive_idx],recall[positive_idx],f1[positive_idx]
        
#     if binary:
#         precision,recall,f1 = precision[0],recall[0],f1[0]
        
    if verbose:
        print(f"precision = {precision}, recall = {recall}, f1 = {f1} (# of datapoints = {len(y_true)})")

    if return_dict:
        return {k:v for k,v in zip(["precision","recall","f1"],[precision,recall,f1])}
    else:
        return precision,recall,f1

    
    
# ----------- probability distributions --------------

# ----- binomial distribution ----
#import scipy

def binomial_probability(sample,n,p):
    """
    Ex: 
    r_values = list(range(n + 1))
    dist = [binom.pmf(r, n, p) for r in r_values ]
    
    """
    return scipy.stats.binom.pmf(sample, n, p)

def binomial_probability_from_samples(samples,n,p,log = True):
    
    probs = np.array([stu.binomial_probability(k,n,p) for k in samples])
    if log:
        return np.sum(np.log(probs))
    else:
        return np.prod(probs)
    
#import sklearn

def roc_curve(
    y_true,
    y_score,
    **kwargs):
    return sklearn.metrics.roc_curve(
        y_true,
        y_score,
        **kwargs)



# ------------ correlations --------------------
def corr(x,y,**kwargs):
    return np.corrcoef(x, y)[1,0]
    
def corr_pearson(x,y,return_p_value = False):
    results = scipy.stats.pearsonr(x, y)
    if return_p_value:
        return results
    else:
        return results[0]
    
def corr_spearman(x,y,return_p_value = False):
    results = scipy.stats.spearmanr(x, y)
    if return_p_value:
        return results
    else:
        return results[0]
    
def corr_kendall(x,y,return_p_value = False):
    results = scipy.stats.kendalltau(x, y)
    if return_p_value:
        return results
    else:
        return results[0]
    
corr_funcs = [
    corr,
    corr_pearson,
    corr_spearman,
    corr_kendall,
]

def correlation_scores_all(
    x,
    y,
    correlation_funcs = None,
    return_dict = True,
    return_p_value= False,
    verbose = False,
    df = None,
    ):
    """
    Purpose: To calculate the correlation
    scores for all the functions
    """
    if df is not None:
        df = df.query(f"({x}=={x}) and ({y}=={y})")
        x = df[x].to_numpy().astype('float')
        y = df[y].to_numpy().astype('float')
    else:
        x = np.array(x).astype('float')
        y = np.array(x).astype('float')
    
    if correlation_funcs is None:
        correlation_funcs = corr_funcs
        
    corr_scores = {k.__name__:k(
        x,y,return_p_value=return_p_value) 
                   for k in correlation_funcs}
    
    if verbose:
        for k,v in corr_scores.items():
            print(f"{k}:{v}")
            
    dic = dict()
    for k,v in corr_scores.items():
        if "float" in str(type(v)):
            dic[k] = dict(correlation = v,pvalue = None)
        elif type(v) == tuple:
            dic[k] = dict(correlation = v[0],pvalue = v[1])
        else:
            dic[k] = dict(correlation = v.correlation,pvalue = v.pvalue)
            
    if not return_dict:
        return list(dic.values())
    return dic
    
    


#from python_tools import statistics_utils as stu



#--- from python_tools ---
from . import numpy_utils as nu
from . import pandas_utils as pu

from . import statistics_utils as stu