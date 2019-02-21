"""Tests for metrics."""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals
import os

import numpy as np
from numpy.testing import assert_almost_equal, assert_equal, assert_raises_regex
import pytest

from scorelib.metrics import (bcubed, conditional_entropy, contingency_matrix,
                              der, goodman_kruskal_tau, mutual_information)
from scorelib.rttm import load_rttm


TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def make_labels(n=1000, n_classes=3, one_hot=False, seed=99999):
    """Return some cluster labelings for a toy dataset."""
    rstate = np.random.RandomState(seed)
    def _one_hot(x):
        classes, x = np.unique(x, return_inverse=True)
        return np.column_stack([x == c for c in classes])
    ref_labels = rstate.randint(n_classes, size=n)
    sys_labels = rstate.randint(n_classes, size=n)
    if one_hot:
        ref_labels = _one_hot(ref_labels)
        sys_labels = _one_hot(sys_labels)
    return ref_labels, sys_labels


def test_contingency_matrix():
    # Test exceptions.
    with assert_raises_regex(
            ValueError, 'ref_labels and sys_labels should either both be 1D'):
        contingency_matrix(np.zeros(5), np.zeros((5, 2)))
    with assert_raises_regex(
            ValueError, 'ref_labels and sys_labels must have same size'):
        contingency_matrix(np.arange(5), np.arange(6))

    # Test 1-D inputs.
    X, Y = make_labels()
    cm = contingency_matrix(X, Y)
    cm_expected = np.array(
        [[106, 114, 117],
         [110, 130, 105],
         [92, 118, 108]])
    assert_equal(cm, cm_expected)

    # Test 2-D inputs.
    X, Y = make_labels(one_hot=True)
    cm = contingency_matrix(X, Y)
    assert_equal(cm, cm_expected)


def test_bcubed():
    x, y = make_labels()

    # Test from inputs.
    p, r, f1 = bcubed(x, y)
    assert_almost_equal(p, 0.3345, 3)
    assert_almost_equal(r, 0.3356, 3)
    assert_almost_equal(f1, 0.3351, 3)

    # Test from CM.
    cm = contingency_matrix(x, y)
    p, r, f1 = bcubed(None, None, cm)
    assert_almost_equal(p, 0.3345, 3)
    assert_almost_equal(r, 0.3356, 3)
    assert_almost_equal(f1, 0.3351, 3)


def test_goodman_kruskal_tau():
    x, y = make_labels()

    # Test from inputs.
    tau_rs, tau_sr = goodman_kruskal_tau(x, y)
    assert_almost_equal(tau_rs, 0.001223, 5)
    assert_almost_equal(tau_sr, 0.001223, 5)

    # Test from CM.
    cm = contingency_matrix(x, y)
    tau_rs, tau_sr = goodman_kruskal_tau(None, None, cm)
    assert_almost_equal(tau_rs, 0.001223, 5)
    assert_almost_equal(tau_sr, 0.001223, 5)


def test_conditional_entropy():
    x, y = make_labels()

    # Test from inputs.
    ce = conditional_entropy(x, y)
    assert_almost_equal(ce, 1.5824, 3)

    # Test from CM.
    cm = contingency_matrix(x, y)
    ce = conditional_entropy(None, None, cm)
    assert_almost_equal(ce, 1.5824, 3)


def test_mutual_information():
    x, y = make_labels()

    # Test from inputs.
    mi, nmi = mutual_information(x, y)
    assert_almost_equal(mi, 0.001767, 5)
    assert_almost_equal(nmi, 0.001116, 5)

    # Test from CM.
    cm = contingency_matrix(x, y)
    mi, nmi = mutual_information(None, None, cm)
    assert_almost_equal(mi, 0.001767, 5)
    assert_almost_equal(nmi, 0.001116, 5)


def test_der():
    ref_turns, _, _ = load_rttm(
        os.path.join(TEST_DIR, 'ref.rttm'))
    sys_turns, _, _ = load_rttm(
        os.path.join(TEST_DIR, 'sys.rttm'))
    expected_der = 26.3931
    file_to_der, global_der = der(ref_turns, sys_turns)
    assert_almost_equal(file_to_der['FILE1'], expected_der, 3)
    assert_almost_equal(global_der, expected_der, 3)
