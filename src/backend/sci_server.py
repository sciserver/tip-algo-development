import itertools
import warnings
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import src.utils.log_time as log_time


def extract_disconnected_auids(
    auid_eids: pd.Series,
    eid_auids: pd.Series,
) -> Iterator[List[int]]:

    distinct_auids = auid_eids.index.values
    auids_to_explore = set(distinct_auids)

    for auid in filter(lambda i: i in auids_to_explore, distinct_auids):
        connected_set = set(list(auid,))

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


def build_adjacency_matrix(
    auid_eids: pd.Series,
    eid_auids: pd.Series,
    auids: List[int],
    weighted: bool = False,
) -> Tuple[np.ndarray, sparse.csr_matrix]:

    # this will be used as the index for assigning the row and column indices
    auids = np.sort(auids)

    # Build the matrix using COO format
    values, row_idxs, col_idxs = list(), list(), list()
    for auid in auids:
        eids = auid_eids[auid]
        co_auids = set(itertools.chain.from_iterable(eid_auids[eids].values))
        co_auids = np.sort(co_auids- set([auid,]))

        idxs = np.searchsorted(auids, co_auids)







# def get_data(year: int) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
#     auid_eids = auid_eid_pairs.groupby("auid")["eid"].apply(list)  # auid -> eids
#     eid_auids = auid_eid_pairs.groupby("eid")["auid"].apply(list)  # eid -> auids

#     return (
#         build_adjacency_matrix(auid_eid_pairs, logger),
#         prior_y,
#         posterior_y,
#     )