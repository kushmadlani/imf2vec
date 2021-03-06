
import math
import numpy as np
import math
import random
from scipy import stats
import scipy.sparse as sp


def m_p_r(R_test, R_hat, verbose=False):
    """(n_users, n_items) returns percentage ranking from two dense matrices"""
    if sp.issparse(R_test):
        R_true = np.array(R_test.todense())
    else:
        R_true = R_test

    (n, m) = R_hat.shape
    R_r = np.zeros(shape=(n,m))

    for i in range(n):
        R_r[i,:] = stats.rankdata(-R_hat[i], "average")/m
        if i%10000==0 and verbose:
            print('processing user {}'.format(i))
    
    r = np.einsum('ij,ij',R_true,R_r)/np.sum(R_true)

    return r

def sparse_to_list(X):
    """list of nonzero indices per row of sparse X"""
    result = np.split(X.indices, X.indptr)[1:-1]
    result = [list(r) for r in result]
    return result

def apk(actual, predicted, k=10):
    """
    Source: https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Source: https://github.com/statisticianinstilettos/recmetrics/blob/master/recmetrics/metrics.py
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def mr(actual, predicted):
    """
    Computes the recall of each user's list of recommendations, and averages precision over all users.
    ----------
    actual : a list of lists
        Actual items to be predicted
        example: [['A', 'B', 'X'], ['A', 'B', 'Y']]
    predicted : a list of lists
        Ordered predictions
        example: [['X', 'Y', 'Z'], ['X', 'Y', 'Z']]
    Returns:
    -------
        recall: int
    """
    def calc_recall(predicted, actual):
        reca = [value for value in predicted if value in actual]
        reca = np.round(float(len(reca)) / float(len(actual)), 4)
        return reca

    recall = np.mean(list(map(calc_recall, predicted, actual)))
    return recall

def evaluate_model(R_hat, test_mat, masked_test_mat, top_n):
    """Given matrix of predicted values calculate MAP@k, Rec@k, MPR on masked + unmasked"""
    top_N_recs = np.array(np.argsort(-R_hat, axis=1)[:,:top_n]).tolist()
    y_true = sparse_to_list(test_mat)    
    
    MAP = mapk(y_true, top_N_recs)
    rec_at_k = mr(y_true, top_N_recs)
    
    del y_true, top_N_recs

    mpr_all = m_p_r(test_mat, R_hat, verbose=False)
    mpr_mask = m_p_r(masked_test_mat, R_hat, verbose=False)

    return MAP, rec_at_k, mpr_all, mpr_mask