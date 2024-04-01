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

"""Test for Sciserver backend SocNL functions."""

import logging
import os

import numpy as np
import pandas as pd
import pytest

import src.backend._sciserver.socnl as ss

def test_default_combine_posterior_prior_y_func():
    """Test the default combine_posterior_prior_y function."""

    arrs = [
        np.array([0.3, 0.4]),
        np.array([0.6, 0.7]),
        np.array([0.9, 1.0]),
    ]

    expected_result = np.array([0.3, 0.4])

    result = ss.default_combine_posterior_prior_y_func(arrs)

    assert np.allclose(result, expected_result)


def test_calculate_prior_y_from_eids_only_seed_serial():
    """Tests the calculate_prior_y_from_eids function."""

    ss.PARALLEL_APPLY = False

    auid_eids = pd.Series(
        index=[1, 2, 3, 4],
        data=[
            [10, 20],
            [20],
            [10],
            [30, 20, 10],
        ],
    )

    eid_score = pd.DataFrame(
        index=[10, 20, 30],
        data=dict(
            seed=[1, 0, 0],
            score=[0.5, 0.30, 0.40],
        ),
    )

    expected = np.array([1.0, 0.0, 1.0, 1.0])

    actual = ss.calculate_prior_y_from_eids(
        auid_eids.index,
        auid_eids,
        eid_score,
        seed_only=True,
    ).values

    assert np.allclose(actual, expected)


def test_calculate_prior_y_from_eids_only_seed_parallel():
    """Tests the calculate_prior_y_from_eids function."""

    ss.PARALLEL_APPLY = True

    auid_eids = pd.Series(
        index=[1, 2, 3, 4],
        data=[
            [10, 20],
            [20],
            [10],
            [30, 20, 10],
        ],
    )

    eid_score = pd.DataFrame(
        index=[10, 20, 30],
        data=dict(
            seed=[1, 0, 0],
            score=[0.5, 0.30, 0.40],
        ),
    )

    expected = np.array([1.0, 0.0, 1.0, 1.0])

    actual = ss.calculate_prior_y_from_eids(
        auid_eids.index,
        auid_eids,
        eid_score,
        seed_only=True,
    ).values

    assert np.allclose(actual, expected)


def test_calculate_prior_y_from_eids_combined_serial():
    """Tests the calculate_prior_y_from_eids function."""

    ss.PARALLEL_APPLY = False

    auid_eids = pd.Series(
        index=[1, 2, 3, 4],
        data=[
            [10, 20],
            [20],
            [10],
            [30, 20, 10],
        ],
    )

    eid_score = pd.DataFrame(
        index=[10, 20, 30],
        data=dict(
            seed=[1, 0, 0],
            score=[0.5, 0.3, 0.4],
        ),
    )

    expected = np.array([0.4, 0.0, 0.5, 0.4])

    actual = ss.calculate_prior_y_from_eids(
        auid_eids.index,
        auid_eids,
        eid_score,
        seed_only=False,
    )

    assert np.allclose(actual.values, expected)


def test_calculate_prior_y_from_eids_combined_parallel():
    """Tests the calculate_prior_y_from_eids function."""

    ss.PARALLEL_APPLY = True

    auid_eids = pd.Series(
        index=[1, 2, 3, 4],
        data=[
            [10, 20],
            [20],
            [10],
            [30, 20, 10],
        ],
    )

    eid_score = pd.DataFrame(
        index=[10, 20, 30],
        data=dict(
            seed=[1, 0, 0],
            score=[0.5, 0.3, 0.4],
        ),
    )

    expected = np.array([0.4, 0.0, 0.5, 0.4])

    actual = ss.calculate_prior_y_from_eids(
        auid_eids.index,
        auid_eids,
        eid_score,
        seed_only=False,
    )

    assert np.allclose(actual, expected)


def test_get_previous_posterior_exists():
    """Tests the get_previous_posterior function."""

    year = 2020

    # this requires some i/o to test
    os.makedirs("./tmp", exist_ok=True)
    ss.POSTERIOR_DATA_PATH = "./tmp/posterior_{year}.parquet"

    # create a dummy file
    (
        pd.Series(
            index=[1, 2, 3, 4],
            data=[0.1, 0.2, 0.3, 0.4],
        )
        .to_frame("score")
        .to_parquet(ss.POSTERIOR_DATA_PATH.format(year=2020))
    )

    test_auids = [1, 2, 5]
    expected = np.array([0.1, 0.2, np.nan])

    actual = ss.get_previous_posterior(test_auids, year)

    np.allclose(actual, expected, equal_nan=True)

    os.remove(ss.POSTERIOR_DATA_PATH.format(year=2020))
    os.rmdir("./tmp")


def test_get_previous_posterior_doesnt_exist():
    """Tests the get_previous_posterior function."""

    year = 2020

    # this requires some i/o to test
    os.makedirs("./tmp", exist_ok=True)
    ss.POSTERIOR_DATA_PATH = "./tmp/posterior_{year}.parquet"

    test_auids = [1, 2, 5]

    actual = ss.get_previous_posterior(test_auids, year)

    os.rmdir("./tmp")

    assert actual is None


def test_update_posterior_nothing_exists():
    """Tests the update_posterior function."""

    auids = [1, 2, 3, 4]
    posterior_y = np.array([0.1, 0.2, 0.3, 0.4])
    year = 2020

    os.makedirs("./tmp", exist_ok=True)
    ss.POSTERIOR_DATA_PATH = "./tmp/posterior_{year}.parquet"

    ss.update_posterior(auids, posterior_y, year, logging.getLogger("test"))

    df = pd.read_parquet(ss.POSTERIOR_DATA_PATH.format(year=2020))

    expected = pd.Series(
        index=[1, 2, 3, 4],
        data=[0.1, 0.2, 0.3, 0.4],
        name="score",
    )

    os.remove(ss.POSTERIOR_DATA_PATH.format(year=2020))
    os.rmdir("./tmp")

    assert df.equals(expected.to_frame("score"))


def test_update_posterior_something_exists():
    """Tests the update_posterior function."""
    ss.POSTERIOR_DATA_PATH = "./tmp/posterior_{year}.parquet"
    os.makedirs("./tmp", exist_ok=True)
    year = 2020

    pd.Series(index=[1, 2, 3], data=[0.9, 0.8, 0.7],).to_frame(
        "score"
    ).to_parquet(ss.POSTERIOR_DATA_PATH.format(year=year))

    auids = [4, 5, 6]
    posterior_y = np.array([0.6, 0.5, 0.4])

    ss.update_posterior(auids, posterior_y, year, logging.getLogger("test"))

    df = pd.read_parquet(ss.POSTERIOR_DATA_PATH.format(year=2020))

    expected = pd.Series(
        index=[1, 2, 3, 4, 5, 6],
        data=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        name="score",
    )

    os.remove(ss.POSTERIOR_DATA_PATH.format(year=year))
    os.rmdir("./tmp")

    assert df.equals(expected.to_frame("score"))
