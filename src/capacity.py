import sys
import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold

def add_filename(values, name, transpose=False):
    """Add a column with the name of the dataset"""
    if transpose:
        name_col = [name for i in values.iloc[:,0]]
        values['Dataset'] = name_col
        return values
    else:
        name_col = [name for i in values]
        q2_df = pd.DataFrame([values, name_col])
        return q2_df

def stat_iter(df):
    """Compute mean and std for each iteration
    
        Parameters:
            df(pd.Dataframe): ['niter', 'score']"""
    df_mean = df.groupby(['Dataset', 'niter']).mean()
    df_std = df.groupby(['Dataset', 'niter']).std()
    df_stats = pd.concat([df_mean, df_std], axis=1)
    df_stats.columns = ['mean', 'std']
    return df_stats

def xgboost_q2(x, y, n_est=10, cross_val=5, depth=20):
    """Return the q2 score of cross-validation using xgboost for regression problems"""

    # create model instance
    bst = XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=0.2)
    # fit model
    kfold = KFold(n_splits=cross_val, shuffle=True)
    scores_xgb = cross_val_score(bst, x, y, cv=kfold, scoring="r2")
    return scores_xgb

def read_csv(filename):
    # Reading datafile with pandas
    # Return HEADER and DATA
    if filename[-3:] != "csv":
        filename += ".csv"
    dataframe = pd.read_csv(filename, header=0)
    dataframe = dataframe.dropna()
    HEADER = dataframe.columns.tolist()
    dataset = dataframe.values
    DATA = np.asarray(dataset[:, :])
    return HEADER, DATA

def read_XY(filename, nY=1, scaling=""):
    # Format data for training
    # Function read_training_data is defined in module (1)
    # if scaling == 'X' X is scaled
    # if scaling == 'Y' Y is scaled
    # if scaling == 'XY' X and Y are scaled
    _, XY = read_csv(filename)

    XY = np.asarray(XY)
    X = XY[:, :-nY]
    Y = XY[:, -nY:]
    X, _, _ = MinMaxScaler(X) if scaling == "X" or scaling == "XY" else X, 0, 0
    Y, _, _ = MinMaxScaler(Y) if scaling == "Y" or scaling == "XY" else Y, 0, 0
    return X, Y

def xgboost_classif(x, y, n_est=10, cross_val=5, depth=20):
    """Classification with XGboost
    return:
    if return_pred:True
    * Acc score: float
    * preds_int: list[int] list of 0/1 xgboost predictions
    * preds_float: list[float] list of xgboost predictions as proba of True
    * y_test: list[int] reference values for computing scores

    else:
    * Acc score: float
    """
    # create model instance
    bst = XGBClassifier(n_estimators=n_est, max_depth=depth, learning_rate=0.2)
    # fit model
    kfold = KFold(n_splits=cross_val, shuffle=True)
    scores_xgb = cross_val_score(bst, x, y, cv=kfold, scoring="accuracy")
    return scores_xgb

def xgboost_from_file(filename, task, cross_val=5, N_ESTIMATOR=30, mdepth=10):
    """Compute XGboost on a given dataset and return the validation score Q2."""
    FILENAME = f"./data/raw/{filename}.csv"
    X_xgb, Y_xgb = read_XY(FILENAME)
    if task == "regression":
        all_xgb_Q2 = xgboost_q2(X_xgb, Y_xgb, cross_val=cross_val, n_est=N_ESTIMATOR, depth=mdepth)
    else:
        all_xgb_Q2 = xgboost_classif(X_xgb, Y_xgb, cross_val=cross_val, n_est=N_ESTIMATOR, depth=mdepth)
    return all_xgb_Q2

def xgb_var_nest(list_est, filename, task, cross_val=5):
    """Load a table of raw data and compute XGboost for all nest in list_est."""
    xgb_scores = []
    list_est_iter = np.array([[i for j in range(cross_val)] for i in list_est]).flatten()
    for n_est in list_est:
            xgb_scores.append(xgboost_from_file(filename, N_ESTIMATOR=n_est,
                                                task=task,
                                                mdepth=1, cross_val=cross_val)) #mdepth is set to 1 to study decision stamp
    xgb_scores = np.array(xgb_scores).flatten()
    df_xgb_score = pd.DataFrame([list_est_iter, xgb_scores]).T
    df_xgb_score.columns=['niter', 'score']
    return df_xgb_score


def xgb_var_nest_multi_ds(list_ds, list_est, task, cross_val=5, save=True):
    """Take a list of dataset and compute XGboost on the raw version of the dataset."""
    ds_all_var_nest = pd.DataFrame()
    for ds in list_ds:
        df_ds = xgb_var_nest(list_est, ds, task, cross_val=cross_val)
        df_ds = add_filename(df_ds, ds, transpose=True)
        ds_all_var_nest = pd.concat([ds_all_var_nest, df_ds], axis=0)
        stat_xgb = stat_iter(ds_all_var_nest)
        stat_xgb = stat_xgb.reset_index()
        stat_xgb['mean'] = pd.to_numeric(stat_xgb['mean'])
    if save:
        stat_xgb.to_csv(f"./data/{task}/xgb_est.csv")
    return stat_xgb

def comparison_xgb(amn_score, xgb_nest):
    """For each score find the needed capacity to reach this score"""
    
    list_est = xgb_nest.index.values
    indice = 0
    while indice<len(list_est)-1 and xgb_nest['mean'][list_est[indice]] < amn_score:
        indice += 1

    return list_est[indice]

def find_capacity_from_stamp(list_ds, amn_result, xgb_stamp, task):
    """For every dataset in list_ds find the capacity necessary to reach score in amn_result
    
        Parameters:
            list_ds(list): Name of the datasets
            amn_results(pd.Dataframe): Score values reached with AMN's framework
            xgb_stamp(pd.Dataframe): XGBoost stamp score for various number of estimator
            regression(Bool): Wether the problem is regression/classification

        Returns:
            result_capacity(pd.Dataframe):Same size as amn_result with the capacity value.         
    """
    score = 'Q2' if task=="regression" else 'Acc'
    result_capacity = {}
    for dataset in list_ds:
        result_capacity[dataset] = []
        xgb_stump_ds = xgb_stamp.loc[xgb_stamp['Dataset']==dataset]
        amn_ds = amn_result.loc[amn_result['Dataset']==dataset]
        for i in amn_ds[score]:
            result_capacity[dataset].append(comparison_xgb(i, xgb_stump_ds))
    return result_capacity

def df_from_dict_capacity(capacity_dict, task, save=True):
    """Create a Dataframe from a dictionary of capacity result {dataset:capacity_score}"""
    df_all = pd.concat([add_filename(capacity_dict[dataset], dataset).T for dataset in capacity_dict])
    df_all.columns = ["capacity", "dataset"]
    if save:
        df_all.to_csv(f'./data/{task}/capacity.csv')
    return df_all

if __name__ == "__main__":
    if sys.argv[1] not in ["regression", "classification"]:
        raise ValueError("First argument must be 'regression' or 'classification'")
    task = sys.argv[1]  # 'regression' or 'classification'
    
    if task=="regression":
        list_ds = ['cpu_activity', 'concrete_compressive_strength', 'energy_efficiency']
        list_est = [i for i in range(100)]
    else:
        list_ds = ['phoneme', 'diabetes', 'wdbc']
        list_est = [nb*10**i for i in range(3) for nb in range(1, 11)]

    stat_xgb = xgb_var_nest_multi_ds(list_ds, list_est, task, cross_val=5)
    ### Nest ###

    stat_xgb = pd.read_csv(f'./data/{task}/xgb_est.csv', index_col=0)
    result_amn = pd.read_csv(f'./data/{task}/method_comparison.csv')
    result_amn = result_amn.loc[result_amn['method']=='AMN']

    dic_capacity = find_capacity_from_stamp(list_ds, result_amn, stat_xgb, task)
    df_reg_capacity = df_from_dict_capacity(dic_capacity, task, save=True)
