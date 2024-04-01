# MIT License

# Copyright (c) 2024 The Johns Hopkins University, Institute for Data Intensive Engineering and Science

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Common functions between algorithms for the SciServer Backend."""

import functools
import itertools
import logging
import os
import warnings
from typing import Callable, Iterable, Iterator, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import src.label_prop.algorithms as algos
import src.utils.log_time as log_time


def extract_disconnected_auids(
    auid_eids: pd.Series,
    eid_auids: pd.Series,
) -> Iterator[List[int]]:
    """Extracts disconnected sets of auids from the given auids and eids.

    Args:
        auid_eids (pd.Series): A pandas series with auids as the index and eids
                               as the values.
        eid_auids (pd.Series): A pandas series with eids as the index and auids
                               as the values.

    Yields:
        Iterator[List[int]]: An iterator with the disconnected sets of auids.
    """
    # sorting to try and make this as deterministic as possible
    distinct_auids = np.sort(auid_eids.index.values)
    auids_to_explore = set(distinct_auids)

    for auid in filter(lambda i: i in auids_to_explore, distinct_auids):
        connected_set = set(
            [
                auid,
            ]
        )

        eids = auid_eids[auid]
        auids = set(itertools.chain.from_iterable(eid_auids[eids].values))

        new = auids - connected_set
        while new:
            connected_set |= new
            eids_to_check = list(
                itertools.chain.from_iterable(auid_eids[list(new)].values)
            )
            connections = set(
                itertools.chain.from_iterable(eid_auids[eids_to_check].values)
            )
            new = connections - connected_set

        auids_to_explore -= connected_set

        yield list(connected_set)


def default_weights_func(x: List[int], _: List[int]) -> List[float]:
    """Default function for calculating the weights for the edges between the auids.

    The default function is to return a list of ones.

    Args:
        x (List[int]): The source auids.
        y (List[int]): The target auids.

    Returns:
        List[float]: A list of ones.
    """

    return np.ones(len(x)).tolist()


def build_adjacency_matrix(
    auid_eids: pd.Series,
    eid_auids: pd.Series,
    auids: List[int],
    weights_f: Callable[[List[int], List[int]], List[float]] = default_weights_func,
    dtype: np.dtype = bool,
) -> Tuple[np.ndarray, sparse.csr_matrix]:
    """Builds an adjacency matrix from the given auids and eids.

    Uses a breadth-first search to find the connected auids and then builds the
    adjacency matrix.

    Args:
        auid_eids (pd.Series): A pandas series with auids as the index and eids
            as the values.
        eid_auids (pd.Series): A pandas series with eids as the index and auids
            as the values.
        auids (List[int]): The auids to build the adjacency matrix from.
        weights_f (Callable[[List[int], List[int]], List[float]]]): A function
            that takes two lists of auids and returns a list of weights for the
            edges between the auids. Defaults to a function that returns a list
            of ones.
        dtype (np.dtype): The data type of the adjacency matrix. Defaults to
            bool.

    Returns:
        Tuple[np.ndarray, sparse.csr_matrix]: A tuple with the auids and the
            adjacency matrix.
    """

    # this will be used as the index for assigning the row and column indices
    auid_idxs = np.argsort(auids)
    auids = auids[auid_idxs]

    # Build the matrix using COO format
    values, row_idxs, col_idxs = list(), list(), list()
    for i, auid in enumerate(auids):
        auid_set = set(
            [
                auid,
            ]
        )
        eids = auid_eids[auid]
        co_auids = set(itertools.chain.from_iterable(eid_auids[eids].values))
        co_auids = np.sort(list(co_auids - auid_set), axis=None)

        col_idxs.extend(np.searchsorted(auids, co_auids).tolist())
        row_idxs.extend([i] * len(co_auids))
        values.extend(weights_f([auid] * len(co_auids), co_auids))

    A = sparse.coo_matrix(
        (values, (row_idxs, col_idxs)),
        shape=(len(auids), len(auids)),
        dtype=dtype,
    ).tocsr()

    return auids, A
