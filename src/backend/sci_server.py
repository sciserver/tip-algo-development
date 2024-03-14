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

import functools
import itertools
import logging
import os
import warnings
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import src.utils.log_time as log_time

def default_combine_posterior_prior_y_func(arrs: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(arrs, axis=1), axis=1)

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

    distinct_auids = auid_eids.index.values
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


def build_adjacency_matrix(
    auid_eids: pd.Series,
    eid_auids: pd.Series,
    auids: List[int],
    weights_f: Optional[Callable[[List[int], List[int]], List[float]]] = lambda x, y: np.ones(len(x)).tolist(),
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
        weights (np.ndarray, optional): The weights to use for the adjacency
            matrix where the weights are in the same order as `auids`. The
            matrix will inherit its type from the weight vector. If None, the
            matrix defaults 1's with values in the matrix being 1/0 with type
            bool. Defaults to None.

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


def calculate_prior_y(
    auids: np.ndarray,
    auid_eids: pd.Series,
    eid_score: pd.Series,
    year: int,
    prior_y_aggregate_eid_score_func: Callable[[np.ndarray], float] = np.mean,
    combine_posterior_prior_y_func: Callable[[List[np.ndarray]], np.ndarray] = default_combine_posterior_prior_y_func,
    posterior_y_missing_value: float = 0.5,
) -> np.ndarray:

    # get all of eids for each auid
    selected_eids = auid_eids[auids]

    prior_y = selected_eids.apply(
        lambda eids: prior_y_aggregate_eid_score_func(eid_score[eids])
    )

    # TODO: support an arbitrary number of years
    posterior_y_path = f"./data/posterior_y_{year}.parquet"
    if os.path.exists(posterior_y_path):
        posterior_y_dframe = pd.read_parquet(posterior_y_path)

        known_auids = posterior_y_dframe.index.values
        new_auids = set(auids) - set(known_auids)
        posterior_y_t_minus_1 = pd.Series(
            data=posterior_y_dframe["score"].values,
            index=known_auids,
        )
        del posterior_y_dframe

        # if there are new ids tat we haven't seen before, we need to add them
        # with the default value.
        if new_auids:
            for auid in new_auids:
                posterior_y_t_minus_1[auid] = posterior_y_missing_value
            posterior_y_t_minus_1.sort_index(inplace=True)

        # There is a chance that there are less auids in the prior_y than in the
        # posterior_y. If that is the case, we need to limit the calculation
        # to the auids that are in both.
        if len(posterior_y_t_minus_1) > 0:
            print(posterior_y_t_minus_1.shape, prior_y.shape)
            print(posterior_y_t_minus_1)
            posterior_matched = posterior_y_t_minus_1.index.intersection(prior_y.index)
            posterior_y_t_minus_1 = posterior_y_t_minus_1[posterior_matched]
            print(prior_y)
            prior_y = combine_posterior_prior_y_func(
                np.stack([prior_y, posterior_y_t_minus_1], axis=1),
            )

    return prior_y


def get_data(
    year: int,
    logger: logging.Logger = None,
    prior_y_aggregate_eid_score_func: Callable[[np.ndarray], float] = np.mean,
    combine_posterior_prior_y_func: Callable[[np.ndarray], np.ndarray] = np.mean,
    adj_mat_dtype: np.dtype = bool,
    numeric_types: np.dtype = np.float32,
    operate_on_subgraphs_separately: bool = False,
) -> Iterable[Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]:

    os.makedirs("./data/cache", exist_ok=True)

    auid_eids = pd.read_parquet(f"./data/auid_eid_{year}.parquet")
    eids = pd.read_parquet(f"./data/eids_{year}.parquet")
    eid_scores = pd.Series(
        data=eids["score"].values,
        index=eids["eid"].values,
    )
    del eids

    with log_time.LogTime(f"Grouping eid-auid by eid for {year}", logger):
        eid_auids = auid_eids.groupby("eid")["auid"].apply(list)  # eid -> auids
    with log_time.LogTime(f"Grouping auid-eid by auid for {year}", logger):
        auid_eids = auid_eids.groupby("auid")["eid"].apply(list)  # auid -> eids

    adj_mat_func = functools.partial(
        build_adjacency_matrix,
        auid_eids,
        eid_auids,
        dtype=adj_mat_dtype,
    )

    if operate_on_subgraphs_separately:
        auids_iter = extract_disconnected_auids(auid_eids, eid_auids)
        #TODO: adding caching for large subgraphs
        auids, A = zip(*map(adj_mat_func, auids_iter))

        prior_y = calculate_prior_y(
            auids,
            auid_eids,
            eid_scores,
            year,
            prior_y_aggregate_eid_score_func,
            combine_posterior_prior_y_func,
        )

        yield A, auids, prior_y
    else:
        # calculating the adjacency matrix for the entire graph can take a long
        # time. We cache the result to avoid recalculating it.
        if not os.path.exists(f"./data/cache/adjacency_matrix_{year}.npz"):
            auids, A = adj_mat_func(auid_eids.index.values)
            sparse.save_npz(f"./data/cache/adjacency_matrix_{year}.npz", A)
            np.save(f"./data/cache/auids_{year}.npy", auids)
        else:
            A = sparse.load_npz(f"./data/cache/adjacency_matrix_{year}.npz")
            auids = np.load(f"./data/cache/auids_{year}.npy")

        prior_y = calculate_prior_y(
            auids,
            auid_eids,
            eid_scores,
            year,
            prior_y_aggregate_eid_score_func,
            combine_posterior_prior_y_func,
        )

        return iter([(A, auids, prior_y)])

def update_posterior(
    auids: np.ndarray,
    posterior_y_values: np.ndarray,
    year: int,
) -> None:

    posterior_path = f"./data/posterior_y_{year}.parquet"
    if os.path.exists(posterior_path):
        existing_posterior_y_df = pd.read_parquet(posterior_path)
        existing_posterior_y = pd.Series(
            data=existing_posterior_y_df["score"].values,
            index=existing_posterior_y_df.index.values,
        )
        # should be safe as we handles auids as a set before this.
        posterior_y = existing_posterior_y.combine_first(
            pd.Series(posterior_y_values, index=auids)
        )
    else:
        posterior_y = pd.Series(posterior_y_values, index=auids)

    # convert series to a dataframe and save to parquet
    posterior_y.to_frame(name="score").to_parquet(f"./data/posterior_y_{year}.parquet")
