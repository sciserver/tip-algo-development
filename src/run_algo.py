import argparse
import itertools
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
import scipy.sparse as sp

import src.backend.elsevier as elsevier
import src.backend.sci_server as sci_server
import src.label_prop.algorithms as algos

MaybeSparseMatrix = Union[np.ndarray, sp.spmatrix]


def run_algo_year(
    algo: algos.Base,
    year: int,
    get_data_func: Callable[[int], Tuple[MaybeSparseMatrix, np.ndarray, np.ndarray]],
    posterior_update_func: Callable[[np.ndarray, np.ndarray, int], None],
) -> np.ndarray:

    for A, auids, prior_y in get_data_func(year):
        posterior_y = algo.fit_predict_graph(A, prior_y)
        posterior_update_func(auids, posterior_y, year)


def run_algo(
    algos: Iterable[algos.Base],
    years: Iterable[int],
    get_data_func: Callable[[int], Tuple[MaybeSparseMatrix, np.ndarray, np.ndarray]],
    posterior_update_func: Callable[[np.ndarray, np.ndarray, int], None],
) -> None:

    for algo, year in zip(algos, years):
        run_algo_year(algo, year, get_data_func, posterior_update_func)


def main(args: Dict[str, Any]):

    runtime = args.get("runtime")

    algo = algos.CAMLP(
        beta=args.get("beta"),
        max_iter=args.get("max_iter"),
        rtol=args.get("rtol"),
        atol=args.get("atol"),
    )

    if runtime == "sciserver":
        get_data_func = sci_server.get_data
        posterior_update_func = sci_server.update_posterior
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
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--runtime", type=str, required=True, default="sciserver")
    parser.add_argument("--beta", type=float, required=True, default=0.1)
    parser.add_argument("--max_iter", type=int, required=True, default=30)
    parser.add_argument("--rtol", type=float, required=True, default=1e-6)
    parser.add_argument("--atol", type=float, required=True, default=1e-6)
    parser.add_argument("--years", type=int, required=True, nargs="+", default=[2010])

    main(args=vars(parser.parse_args()))
