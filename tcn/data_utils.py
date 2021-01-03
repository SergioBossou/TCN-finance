# ======================================================================================================================
# title            : 
# description      : 
# author           : Sergio
# date             : 19/12/2020
# ======================================================================================================================

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler





def concatin(what, colnames=None, **kwargs):

    """**Concat a list of pandas.DataFrame along axis 1 on indexes intersection**

    Parameters
    ----------
    what : list
        list of pandas.DataFrame
    colnames : list
        list of new columns names

    Returns
    -------
    df : pandas.DataFrame
        Resulting DataFrame

    """
    if colnames is None:
        return pd.concat(what, axis=1, join='inner', **kwargs).dropna()

    else:
        xc = pd.concat(what, axis=1, join='inner', **kwargs).dropna()
        xc.columns = colnames

        return xc



def preprocess(data):
    """ **Min max scaling of data**

    Parameters
    ----------
    data: pd.DataFrame
        Data to be scaled

    Returns
    -------
     scaled_data: pd.DataFrame
        Min-Max scaled data

    """
    scaler = MinMaxScaler()
    scaled_data = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
    return scaled_data


def data_in_shape(data, n_h, n_x, n_t, rolling_window=1, device='cpu'):
    """

    Parameters
    ----------
    data: pd.DataFrame
    n_h :  int
        Receptive field
    n_x : int
        number of features of data
    n_t : int
        number of days in the future we want to predict
    rolling_window : int
        Number of step between two windows

    Returns
    -------
    Returns encod_sample and decod_sample that are used respectively by encoder and decoder.
    encod_sample : pd.DataFrame
        Shape (-1, n_x, n_h). It is composed of receptive fields separated by the rolling window
    decod_sample: pd.DataFrame
        Shape (-1, n_x, n_t). It is composed of the future data that correspond to each receptive field
    """
    end = n_t + n_h + ((len(data) - n_t - n_h) // rolling_window) * rolling_window

    data = data.iloc[: end, :]

    n_sample = ((len(data) - n_t - n_h) // rolling_window) + 1

    decod_sample = np.full((n_sample, n_x, n_t), None, dtype='float')
    encod_sample = np.full((n_sample, n_x, n_h), None, dtype='float')

    for j in range(n_sample):
        decod_sample[j, :, :] = data.iloc[rolling_window * j + n_h: rolling_window * j + n_h + n_t, :].T
        encod_sample[j, :, :] = data.iloc[rolling_window * j: rolling_window * j + n_h, :].T

    decod_sample = torch.from_numpy(decod_sample).float().to(device)
    encod_sample = torch.from_numpy(encod_sample).float().to(device)


    return decod_sample, encod_sample


def concatenate_time(list_of_data, device):
    """         Concatenates dataframes in list_of_data along the last dimension


    Parameters
    ----------
    list_of_data: list of pd.DataFrame of 3 dimensions
    device:
        cuda or cpu

    Returns
    -------
    result: torch.tensor
    """
    n, m, p = list_of_data[0].shape[0], list_of_data[0].shape[1], len(list_of_data)
    result = torch.full((n, m, p), np.nan).float().to(device)
    for i in range(p):
        result[:, :, i:i + 1] = list_of_data[i]
    return result


