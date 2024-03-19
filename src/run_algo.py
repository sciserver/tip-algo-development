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

import argparse
import functools
import itertools
import logging
import os
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
import scipy.sparse as sp

import src.backend.elsevier as elsevier
import src.backend.sciserver as sciserver
import src.label_prop.algorithms as algos
import src.utils.log_time as log_time

MaybeSparseMatrix = Union[np.ndarray, sp.spmatrix]


def run_algo_year(
    algo_instances: Iterable[algos.Base],
    year: int,
    get_data_func: Callable[
        [int, logging.Logger], Tuple[MaybeSparseMatrix, np.ndarray, np.ndarray]
    ],
    posterior_update_func: Callable[[np.ndarray, np.ndarray, int], None],
    log_dir: str,
) -> np.ndarray:
    """Run the given algorithm instances for the given year.


    Args:
        algo_instances (Iterable[algos.Base]): An iterable with the algorithm
            instances to run.
        year (int): The year to run the algorithm instances for.
        get_data_func (Callable[[int, logging.Logger], Tuple[MaybeSparseMatrix, np.ndarray, np.ndarray]]): A function that returns the data for the given year.
        posterior_update_func (Callable[[np.ndarray, np.ndarray, int], None]): A function that updates the posterior for the given year.
        log_dir (str): The directory to save the logs to.

    Returns:
        np.ndarray: The posterior for the given year.

    """

    if log_dir:
        logger = log_time.setup_logger(
            f"run_algo_year_{year}", f"{log_dir}/run_algo_year_{year}.log"
        )
    else:
        logger = log_time.PrintLogger(f"run_algo_year_{year}")

    for i, (algo, (A, auids, prior_y)) in enumerate(
        zip(algo_instances, get_data_func(year, logger)), start=1
    ):
        print(i, A.shape, auids.shape, prior_y.shape)
        with log_time.LogTime(f"Fitting data for {year}, ajd matrix {i}", logger):
            posterior_y = algo.fit_predict_graph(A, prior_y)
        with log_time.LogTime(f"Updating posterior for {year}", logger):
            posterior_update_func(auids, posterior_y, year)


def main(args: Dict[str, Any]):
    """Run the algorithm for the given arguments."""

    os.makedirs(args.get("log_dir"), exist_ok=True)

    runtime = args.get("runtime")

    algo = algos.CAMLP(
        beta=args.get("beta"),
        max_iter=args.get("max_iter"),
        rtol=args.get("rtol"),
        atol=args.get("atol"),
    )

    if runtime == "sciserver":
        get_data_func = functools.partial(
            sciserver.get_data,
            prior_y_aggregate_eid_score_func=np.mean,
            combine_posterior_prior_y_func=functools.partial(np.mean, axis=1),
            operate_on_subgraphs_separately=args.get("parse_subgraphs_separately"),
        )
        posterior_update_func = sciserver.update_posterior
    elif runtime == "elsevier":
        get_data_func = elsevier.get_data
        posterior_update_func = elsevier.update_posterior
    else:
        raise ValueError(f"Runtime {runtime} not supported")

    for year in args.get("years"):
        run_algo_year(
            itertools.repeat(algo),
            year,
            get_data_func=get_data_func,
            posterior_update_func=posterior_update_func,
            log_dir=args.get("log_dir"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--runtime", type=str, default="sciserver")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[
            2010,
            2011,
        ],
    )
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument(
        "--parse_subgraphs_separately",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    main(args=vars(parser.parse_args()))
