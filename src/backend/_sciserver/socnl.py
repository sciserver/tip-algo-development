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

"""The SciServer backend for SocNL data handling."""

import functools
import logging
import os
import warnings
from typing import Callable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import scipy.sparse as sparse


try:
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True)
    PARALLEL_APPLY = True
except ImportError: # pragma: no cover
    warnings.warn(
        "pandarallel not installed, parallel processing will not be available."
    )
    PARALLEL_APPLY = False

import src.backend._sciserver.common as ss_common
import src.utils.log_time as log_time

MIN_ARR_SIZE_FOR_CACHE = 10_000
POSTERIOR_DATA_PATH = "./data/socnl/posterior_y_{year}.parquet"
PRIOR_Y_DATA_PATH = "./data/socnl/prior_y_{year}.json"


def default_combine_posterior_prior_y_func(arrs: Iterable[np.ndarray]) -> np.ndarray:
    """Default function for combining the posterior for years t-1..t-n and the prior for year t.

    The default behavior is ignore the posterior values and only use the prior for the current year.

    The function should return 0s for any auids that should predicted on.

    Args:
        arrs (List[np.ndarray]): A list of arrays to combine. The arrays are always
            ordered from the newset to the oldest.The first array is the prior
            for the current year.

    Returns:
        np.ndarray: The combined array.

    """

    return arrs[0]


def calculate_prior_y_from_eids(
    auids: pd.Series,
    auid_eids: pd.Series,  # auid -> eids:List[int]
    eids_vals: pd.DataFrame,  # eid:int -> [score:float, seed:int]
    seed_only: bool = True,
    agg_score_func: Callable[[np.ndarray], float] = np.mean,
) -> pd.Series:  # auid -> float
    """Calculate the prior using the eids score values.

    Zeroed valued auids are not labeled.

    """

    selected_eids = auid_eids[auids]

    if len(selected_eids) > MIN_ARR_SIZE_FOR_CACHE or PARALLEL_APPLY:
        is_seed_author = selected_eids.parallel_apply(
            lambda eids: np.any(eids_vals.loc[eids, "seed"].values)
        )

        auid_score = selected_eids.parallel_apply(
            lambda eids: agg_score_func(eids_vals.loc[eids, "score"].values)
        )
    else:
        is_seed_author = selected_eids.apply(
            lambda eids: np.any(eids_vals.loc[eids, "seed"].values)
        )

        auid_score = selected_eids.apply(
            lambda eids: agg_score_func(eids_vals.loc[eids, "score"].values)
        )

    if seed_only:
        y = is_seed_author
    else:
        y = auid_score * is_seed_author

    return pd.Series(index=selected_eids.index.values, data=y)


def get_previous_posterior(auids: np.ndarray, year: int) -> Union[np.ndarray, None]:
    """Get the posterior for the given year.

    If the an auid does not exists in the posterior for a year then is will
    be assiged np.nan

    Args:
        auids (np.ndarray): The auids to get the posterior for.
        year (int): The year to get the posterior for.

    Returns:
        np.ndarray: The posterior for the given year.
    """
    if not os.path.exists(POSTERIOR_DATA_PATH.format(year=year)):
        return None

    post_df = pd.read_parquet(POSTERIOR_DATA_PATH.format(year=year))

    post_s = pd.Series(index=post_df.index.values, data=post_df["score"].values)

    return post_s.reindex(auids).values


def get_data(
    year: int,
    logger: logging.Logger,
    input_prior: np.ndarray,
    numeric_types: np.dtype = np.float32,
    adj_mat_dtype: np.dtype = bool,
    operate_on_subgraphs_separately: bool = False,
    prior_y_aggregate_eid_score_func: Callable[[np.ndarray], float] = np.mean,
    seed_only: bool = True,
    n_years_lookback: int = 1,
    combine_posterior_prior_y_func: Callable[
        [Iterable[np.ndarray]], np.ndarray
    ] = default_combine_posterior_prior_y_func,
) -> Iterable[Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]:
    """Get the data for the given year.

    Args:
        year (int): The year to get the data for.
        logger (logging.Logger): The logger to use.

    Yields:
        Iterable[Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]: An iterable
            containing tuples with an adjacency matrix, the auids, and the prior
            for the given year. If opertate_on_subgraphs_separately is True then
            the adjacency matrix will be for a disconnected subgraph. If
            operate_on_subgraphs_separately is False then the iterable contains
            a single tuple representing the entire graph.
    """

    # If we want to operate on subgraphs in with SocNL we need to make decisions
    # about what to do with subgraphs that have no labelled nodes.
    if operate_on_subgraphs_separately:
        raise NotImplementedError(
            "operate_on_subgraphs_separately is not implemented for SocNL."
        )

    if input_prior.sum() != 1:
        raise ValueError("The prior must sum to 1.")

    if input_prior.ndim > 1:
        raise ValueError("The prior must be 1D.")

    os.makedirs("./data/socnl/cache", exist_ok=True)

    auid_eids = pd.read_parquet(f"./data/auid_eid_{year}.parquet")
    eids = pd.read_parquet(f"./data/eids_{year}.parquet")
    numeric_types = numeric_types if numeric_types else eids["score"].dtype

    with log_time.LogTime(f"Grouping eid-auid by eid for {year}", logger):
        eid_auids = auid_eids.groupby("eid")["auid"].apply(list)  # eid -> auids
    with log_time.LogTime(f"Grouping auid-eid by auid for {year}", logger):
        auid_eids = auid_eids.groupby("auid")["eid"].apply(list)  # auid -> eids

    auids = auid_eids.index.values

    # different from CAMLP we need to know the author labels before we build
    # the adjacency matrix, so we can order it correctly.
    with log_time.LogTime(f"Building prior for {year}", logger):
        prior_y = calculate_prior_y_from_eids(
            auids,
            auid_eids,
            eids,
            seed_only,
            prior_y_aggregate_eid_score_func,
        )

    # To ensure we account for previous years authors that we would like to pass
    # we need to load the posterior values for previous years. After which we
    # sort.
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
        combined_y_values = combine_posterior_prior_y_func(
            [prior_y.values] + previous_posteriors
        )

    combined_y = pd.Series(index=prior_y.index.values, data=combined_y_values)

    # sort the auids by the prior values such that the labled values appear
    # first in the adjacency matrix
    sort_idx = np.argsort(-1 * combined_y.values)
    auids = combined_y.index.values[sort_idx]
    sorted_combined_ys = combined_y.values[sort_idx]
    labelled_ys = sorted_combined_ys[sorted_combined_ys > 0]
    auid_eids = auid_eids.loc[auids]

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
                f"./data/socnl/cache/{'iter_'*operate_on_subgraphs_separately}adjacency_matrix_{year}_{i}.npz"
            ):
                logger.info("Found cached adjacency matrix, loading...")
                with log_time.LogTime(f"Loading adjacency matrix {i}", logger):
                    A = sparse.load_npz(
                        f"./data/socnl/cache/{'iter_'*operate_on_subgraphs_separately}adjacency_matrix_{year}_{i}.npz"
                    )
                    auids = np.load(
                        f"./data/socnl/cache/{'iter_'*operate_on_subgraphs_separately}auids_{year}_{i}.npy"
                    )
            else:
                logger.info("No cached adjacency matrix found, building...")
                with log_time.LogTime(f"Building adjacency matrix {i}", logger):
                    auids, A = adj_mat_func(auids)
                with log_time.LogTime(f"Caching adjacency matrix {i}", logger):
                    logger.info(f"Saving adjacency matrix to cache")
                    sparse.save_npz(
                        f"./data/socnl/cache/{'iter_'*operate_on_subgraphs_separately}adjacency_matrix_{year}_{i}.npz",
                        A,
                    )
                    logger.info(f"Saving auids to cache")
                    np.save(
                        f"./data/socnl/cache/{'iter_'*operate_on_subgraphs_separately}auids_{year}_{i}.npy",
                        auids,
                    )
        else:
            with log_time.LogTime(f"Building adjacency matrix {i}", logger):
                auids, A = adj_mat_func(auids)

        # make y multidimensional and append prior to the bottom
        y = np.stack([1 - labelled_ys, labelled_ys], axis=1)
        y = np.concatenate([y, np.atleast_2d(input_prior)])

        yield A, auids, labelled_ys.astype(numeric_types)


def update_posterior(
    auids: np.ndarray,
    posterior_y_values: np.ndarray,
    year: int,
    logger: logging.Logger,
) -> None:

    posterior_path = POSTERIOR_DATA_PATH.format(year=year)
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
