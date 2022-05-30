import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd


def variance(allocations, covariance_matrix):
    return np.dot(np.matmul(covariance_matrix, allocations), allocations)


def linkage(return_series: pd.DataFrame):
    distance = ((1 - return_series.corr()) / 2) ** 0.5
    return sch.linkage(ssd.squareform(distance), "single")

    
def quasi_diagonalize(link, matrix: pd.DataFrame):
    leaves = sch.leaves_list(link)
    diagonal = [matrix.index[l] for l in leaves]
    return matrix.reindex(diagonal)[diagonal]


def ivp_allocation(covariance_matrix: pd.DataFrame):
    allocations = 1 / np.diag(covariance_matrix)
    allocations /= allocations.sum()
    return pd.Series(allocations, index=covariance_matrix.index)


def hrp_allocation(covariance_matrix: pd.DataFrame):
    if len(covariance_matrix.index) <= 2:
        return ivp_allocation(covariance_matrix)

    index = int(len(covariance_matrix.index) / 2)
    left_cluster = covariance_matrix.index[:index]
    left_cluster = covariance_matrix.loc[left_cluster, left_cluster]
    right_cluster = covariance_matrix.index[index:]
    right_cluster = covariance_matrix.loc[right_cluster, right_cluster]

    left_allocation = hrp_allocation(left_cluster)
    right_allocation = hrp_allocation(right_cluster)

    left_variance = variance(ivp_allocation(left_cluster), left_cluster)
    right_variance = variance(ivp_allocation(right_cluster), right_cluster)

    left_weight = right_variance / (left_variance + right_variance)

    return pd.concat(
        [left_weight * left_allocation, (1 - left_weight) * right_allocation]
    )
