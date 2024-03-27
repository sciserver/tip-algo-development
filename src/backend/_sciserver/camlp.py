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

"""CAMLP SciServer backend implementation."""

import functools
import logging
import os
import warnings
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse

import src.backend._sciserver.common as ss_common

try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    PARALLEL_APPLY = True
except ImportError:
    warnings.warn(
        "pandarallel not installed, parallel processing will not be available."
    )
    PARALLEL_APPLY = False

import src.utils.log_time as log_time

MIN_ARR_SIZE_FOR_CACHE = 10_000
POSTIEOR_DATA_PATH = "./data/camlp/posterior_y_{year}.parquet"


def default_combine_posterior_prior_y_func(arrs: List[np.ndarray]) -> np.ndarray:
    """Default function for combining the posterior for years t-1..t-n and the prior for year t.

    The default function is to take the mean of the posterior and prior values.

    Args:
        arrs (List[np.ndarray]): A list of arrays to combine. The arrays are always
            ordered from the newest to the oldest. The first array is the prior
            for the current year.

    Returns:
        np.ndarray: The combined array.

    """

    if any(arr.ndim > 1 for arr in arrs):
        raise ValueError("All arrays must be 1D.")

    length = arrs[0].shape[0]
    if not all(arr.shape[0] == length for arr in arrs):
        raise ValueError("All arrays must be same length.")

    return np.nanmean(np.stack(arrs, axis=1), axis=1)


def calculate_prior_y_from_eids(
    auids: np.ndarray,
    auid_eids: pd.Series,  # auid:int -> eids:List[int]
    eid_score: pd.Series,  # eid:int -> score:float
    agg_score_func: Callable[[np.ndarray], float] = np.mean,
) -> np.ndarray:

    selected_eids = auid_eids[auids]

    if len(selected_eids) > MIN_ARR_SIZE_FOR_CACHE or PARALLEL_APPLY:
        y = selected_eids.parallel_apply(
            lambda eids: agg_score_func(eid_score[eids])
        ).astype(eid_score.dtype)
    else:
        y = selected_eids.apply(lambda eids: agg_score_func(eid_score[eids])).astype(
            eid_score.dtype
        )

    return y


def get_previous_posterior(
    auids: np.ndarray,
    year: int,
) -> Union[np.ndarray, None]:
    """Get the previous posterior for the given auids.

    If the an auid does not exists in the posterior for a year then is will
    be assiged np.nan

    Args:
        auids (np.ndarray): The auids to get the previous posterior for.
        year (int): The year to get the previous posterior for.

    Returns:
        np.ndarray: The previous posterior for the given auids if the posterior
            exists, otherwise None.
    """

    if not os.path.exists(POSTIEOR_DATA_PATH.format(year=year)):
        return None

    post_df = pd.read_parquet(POSTIEOR_DATA_PATH.format(year=year))
    post_s = pd.Series(
        data=post_df["score"].values,
        index=post_df.index.values,
    )
    # reindexing using the target auids will get the posterior values in the
    # same order as the target auids and any auids that are not in the posterior
    # will be assigned np.nan
    return post_s.reindex(auids).values


def get_data(
    year: int,
    logger: logging.Logger = None,
    prior_y_aggregate_eid_score_func: Callable[[np.ndarray], float] = np.mean,
    n_years_lookback: int = 1,
    combine_posterior_prior_y_func: Callable[
        [List[np.ndarray]], np.ndarray
    ] = default_combine_posterior_prior_y_func,
    adj_mat_dtype: np.dtype = bool,
    numeric_types: np.dtype = np.float32,
    operate_on_subgraphs_separately: bool = False,
) -> Iterable[Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]:
    """The get_data function for the SciServer backend.

    Args:
        year (int): The year to get the data for.
        algorithm_name (str): The name of the algorithm to get the data for.
        logger (logging.Logger): The logger to use. Defaults to None.
        prior_y_aggregate_eid_score_func (Callable[[np.ndarray], float]): A function
            that takes an array of scores and returns a single score. Defaults to
            np.mean.
        n_years_lookback (int): The number of years to look back when getting the
            previous posterior. Defaults to 1.
        combine_posterior_prior_y_func (Callable[[List[np.ndarray]], np.ndarray]): A
            function that takes a list of arrays and returns a single array. Defaults
            to default_combine_posterior_prior_y_func.
        adj_mat_dtype (np.dtype): The data type of the adjacency matrix. Defaults to
            bool.
        numeric_types (np.dtype): The data type of the numeric values. Defaults to
            np.float32.
        operate_on_subgraphs_separately (bool): Whether to operate on subgraphs
            separately. Defaults to False.

    Yields:
        Iterable[Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]: An iterable
            containing tuples with an adjacency matrix, the auids, and the prior
            for the given year. If opertate_on_subgraphs_separately is True then
            the adjacency matrix will be for a disconnected subgraph. If
            operate_on_subgraphs_separately is False then the iterable contains
            a single tuple representing the entire graph.

    """

    os.makedirs("./data/camlp/cache", exist_ok=True)

    auid_eids = pd.read_parquet(f"./data/auid_eid_{year}.parquet")
    eids = pd.read_parquet(f"./data/eids_{year}.parquet")
    numberic_types = numeric_types if numeric_types else eids["score"].dtype

    eid_score = pd.Series(
        data=eids["score"].values.astype(numberic_types),
        index=eids["eid"].values,
    )

    del eids

    with log_time.LogTime(f"Grouping eid-auid by eid for {year}", logger):
        eid_auids = auid_eids.groupby("eid")["auid"].apply(list)  # eid -> auids
    with log_time.LogTime(f"Grouping auid-eid by auid for {year}", logger):
        auid_eids = auid_eids.groupby("auid")["eid"].apply(list)  # auid -> eids

    adj_mat_func = functools.partial(
        ss_common.build_adjacency_matrix,
        auid_eids,
        eid_auids,
        dtype=adj_mat_dtype,
    )

    all_auids = auid_eids.index.values

    auids_iter = (
        ss_common.extract_disconnected_auids(auid_eids, eid_auids)
        if operate_on_subgraphs_separately
        else [all_auids]
    )

    for i, auids in enumerate(auids_iter):

        if len(auids) > MIN_ARR_SIZE_FOR_CACHE:
            # TODO: This section might be too hard to read
            logger.info(f"n auids: {len(auids)}, looking for cached adjacency matrix")
            if os.path.exists(
                f"./data/cache/{'iter_'*operate_on_subgraphs_separately}adjacency_matrix_{year}_{i}.npz"
            ):
                logger.info("Found cached adjacency matrix, loading...")
                with log_time.LogTime(f"Loading adjacency matrix {i}", logger):
                    A = sparse.load_npz(
                        f"./data/camlp/cache/{'iter_'*operate_on_subgraphs_separately}adjacency_matrix_{year}_{i}.npz"
                    )
                    auids = np.load(
                        f"./data/camlp/cache/{'iter_'*operate_on_subgraphs_separately}auids_{year}_{i}.npy"
                    )
            else:
                logger.info("No cached adjacency matrix found, building...")
                with log_time.LogTime(f"Building adjacency matrix {i}", logger):
                    auids, A = adj_mat_func(auids)
                with log_time.LogTime(f"Caching adjacency matrix {i}", logger):
                    logger.info(f"Saving adjacency matrix to cache")
                    sparse.save_npz(
                        f"./data/camlp/cache/{'iter_'*operate_on_subgraphs_separately}adjacency_matrix_{year}_{i}.npz",
                        A,
                    )
                    logger.info(f"Saving auids to cache")
                    np.save(
                        f"./data/camlp/cache/{'iter_'*operate_on_subgraphs_separately}auids_{year}_{i}.npy",
                        auids,
                    )
        else:
            with log_time.LogTime(f"Building adjacency matrix {i}", logger):
                auids, A = adj_mat_func(auids)

        with log_time.LogTime(f"Calculating prior_y  for {year} from eids", logger):
            prior_y_eids = calculate_prior_y_from_eids(
                auids,
                auid_eids,
                eid_score,
                prior_y_aggregate_eid_score_func,
            )

        with log_time.LogTime(
            f"Retrieving posteriors for previous {n_years_lookback} years", logger
        ):
            previous_posteriors = list(
                filter(
                    lambda x: x is not None,
                    map(
                        lambda year: get_previous_posterior(auids, year),
                        range(year - 1, year - n_years_lookback - 1, -1),
                    ),
                )
            )

        with log_time.LogTime(
            f"Combining posteriors for previous {n_years_lookback} years", logger
        ):
            prior_y = combine_posterior_prior_y_func(
                [prior_y_eids] + previous_posteriors
            )

        yield A, auids, prior_y.astype(numeric_types)


def update_posterior(
    auids: np.ndarray,
    posterior_y_values: np.ndarray,
    year: int,
    logger: logging.Logger,
) -> None:

    posterior_path = POSTIEOR_DATA_PATH.format(year=year)
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
    posterior_y.to_frame(name="score").to_parquet(posterior_path)
