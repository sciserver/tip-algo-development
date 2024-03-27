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

"""The SciServer backend for the data handling."""

import functools
import logging
import warnings
from typing import Any, Callable, Dict, Tuple

import numpy as np
import scipy.sparse as sparse

import src.label_prop.algorithms as algos
import src.backend._sciserver.camlp as ss_camlp
import src.backend._sciserver.socnl as ss_socnl


def select_get_data_func(
    algorithm: algos.TipGraphAlgorithm,
    adj_mat_dtype: np.dtype = bool,
    numeric_types: np.dtype = np.float32,
    get_data_kwargs: Dict[str, Any] = {},
) -> Callable[[int, logging.Logger], Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]]:

    if str(algorithm) == algos.CAMLP.NAME:
        return functools.partial(
            ss_camlp.get_data,
            prior_y_aggregate_eid_score_func=get_data_kwargs.get(
                "prior_y_aggregate_eid_score_func", np.mean
            ),
            combine_posterior_prior_y_func=get_data_kwargs.get(
                "combine_posterior_prior_y_func",
                ss_camlp.default_combine_posterior_prior_y_func,
            ),
            adj_mat_dtype=adj_mat_dtype,
            numeric_types=numeric_types,
            operate_on_subgraphs_separately=get_data_kwargs.get(
                "operate_on_subgraphs_separately", False
            ),
        )
    elif str(algorithm) == algos.SocNL.NAME:
        prior_str = get_data_kwargs.get("prior")
        if prior_str is None:
            raise ValueError("Prior must be specified for SocNL.")

        input_prior = np.array(
            list(map(float, prior_str.strip().split(","))),
            dtype=numeric_types,
        )

        return functools.partial(
            ss_socnl.get_data,
            input_prior=input_prior,
            prior_y_aggregate_eid_score_func=np.mean,
            combine_posterior_prior_y_func=ss_camlp.default_combine_posterior_prior_y_func,
            adj_mat_dtype=adj_mat_dtype,
            numeric_types=numeric_types,
            operate_on_subgraphs_separately=False,
        )
    else:
        raise ValueError(f"Algorithm {str(algorithm)} not supported")


def select_update_posterior_func(
    algorithm: algos.TipGraphAlgorithm,
) -> Callable[[np.ndarray, np.ndarray, int, logging.Logger], None]:

    if str(algorithm) == algos.CAMLP.NAME:
        return ss_camlp.update_posterior
    elif str(algorithm) == algos.SocNL.NAME:
        return ss_socnl.update_posterior
    else:
        raise ValueError(f"Algorithm {str(algorithm)} not supported")
