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

import itertools
import logging
import os
import warnings
from typing import Callable, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse

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
    weights: Optional[np.ndarray] = None,
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
    auids = np.sort(auids)

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
        values.extend([True] * len(co_auids))

    A = sparse.coo_matrix(
        (values, (row_idxs, col_idxs)),
        shape=(len(auids), len(auids)),
        dtype=bool,
    ).tocsr()

    return auids, A


def calculate_prior_y(
    auids: np.ndarray,
    auid_eids: pd.Series,
    eid_score: pd.Series,
    year: int,
    prior_y_aggregate_eid_score_func: Callable[[np.ndarray], float] = np.mean,
    combine_posterior_prior_y_func: Callable[[np.ndarray], np.ndarray] = lambda arrs: np.mean(arrs, axis=1),
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

        if new_auids:
            for auid in new_auids:
                posterior_y_t_minus_1[auid] = posterior_y_missing_value
            posterior_y_t_minus_1.sort_index(inplace=True)

        if len(posterior_y_t_minus_1) > 0:
            print(posterior_y_t_minus_1.shape, prior_y.shape)
            print(posterior_y_t_minus_1)
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
) -> Iterable[Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]:
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

    for i, connected_auids in enumerate(
        extract_disconnected_auids(auid_eids, eid_auids), start=1
    ):
        with log_time.LogTime(
            f"Building adjacency matrix for {year}, disconnected set {i}", logger
        ):
            auids, A = build_adjacency_matrix(
                auid_eids,
                eid_auids,
                connected_auids,
                False,
            )

        with log_time.LogTime(
            f"Calculating prior_y for {year}, disconnected set {i}", logger
        ):
            prior_y = calculate_prior_y(
                auids,
                auid_eids,
                eid_scores,
                year,
                prior_y_aggregate_eid_score_func,
                combine_posterior_prior_y_func,
            )

        yield A, auids, prior_y


def update_posterior(
    auids: np.ndarray,
    posterior_y_values: np.ndarray,
    year: int,
) -> None:

    posterior_path = f"./data/posterior_y_{year}.parquet"
    if os.path.exists(posterior_path):
        existing_posterior_y = pd.read_parquet(posterior_path)
        # should be safe as we handles auids as a set before this.
        posterior_y = existing_posterior_y.combine_first(
            pd.Series(posterior_y_values, index=auids)
        )
    else:
        posterior_y = pd.Series(posterior_y_values, index=auids)

    # convert series to a dataframe and save to parquet
    posterior_y.to_frame(name="score").to_parquet(f"./data/posterior_y_{year}.parquet")
