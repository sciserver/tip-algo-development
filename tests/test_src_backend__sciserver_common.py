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

"""Test for SciServer backend common functions."""

import numpy as np
import pandas as pd

import src.backend._sciserver.common as ss


def test_default_weights_func():
    """Test the default weights function."""

    x = [1] * 5
    y = [1] * 5

    expected = [1.0] * len(x)
    actual = ss.default_weights_func(x, y)

    assert np.allclose(expected, actual)


def test_extract_disconnected_auids():
    """Tests the extract_disconnected_auids function."""

    auid_eids = pd.Series(
        index=[1, 2, 3, 4],
        data=[
            [10, 20],
            [20],
            [10],
            [30],
        ],
    )
    eid_auids = pd.Series(
        index=[10, 20, 30],
        data=[
            [1, 3],
            [1, 2],
            [4],
        ],
    )

    expected_result = [set([1, 2, 3]), set([4])]

    result = list(map(set, ss.extract_disconnected_auids(auid_eids, eid_auids)))

    assert result == expected_result


def test_build_adjacency_matrix():
    """Tests the build_adjacency_matrix function."""

    auid_eids = pd.Series(
        index=[1, 2, 3, 4],
        data=[
            [10, 20],
            [20],
            [10],
            [30],
        ],
    )
    eid_auids = pd.Series(
        index=[10, 20, 30],
        data=[
            [1, 3],
            [1, 2],
            [4],
        ],
    )

    expected_result = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    result_auids, result_A = ss.build_adjacency_matrix(
        auid_eids,
        eid_auids,
        auid_eids.index,
    )

    assert np.allclose(result_A.todense(), expected_result)
    assert np.allclose(result_auids, auid_eids.index)
