# -*- coding: utf-8 -*-
"""
Unit tests for recompute initital states probabilities.

"""
from decimal import Decimal as d
import unittest

import numpy as np

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class RecomputeInitialStatesTestCase(BaseTestCase):
    def smoke_test_recompute_initial_states(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(1).ln(), d(2).ln()],
            [d(1).ln(), d(2).ln()]])
        result = self.model._recompute_initial_states(log_gamma)
        expected_result = np.array(
            [d(1), d(1), d(1)])
        np.testing.assert_array_equal(result, expected_result)

        log_gamma = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()],
            [d(1).ln(), d(2).ln(), d(3).ln()]])
        result = self.model._recompute_initial_states(log_gamma)
        expected_result = np.array(
            [d(1), d(1)])
        np.testing.assert_array_equal(result, expected_result)

    def test_empty_loggamma(self):
        log_gamma = np.empty((0, 0))
        with self.assertRaises(ValueError):
            self.model._recompute_initial_states(log_gamma)

    def test_none_loggamma(self):
        log_gamma = None
        with self.assertRaises(TypeError):
            self.model._recompute_initial_states(log_gamma)

    def test_invalid_size_loggamma(self):
        log_gamma = np.ones((1, 1, 1))
        with self.assertRaises(ValueError):
            self.model._recompute_initial_states(log_gamma)

    def test_one_state(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()]])
        result = self.model._recompute_initial_states(log_gamma)
        expected_result = np.array(
            [d(1)])
        np.testing.assert_array_equal(result, expected_result)

    def test_one_time(self):
        log_gamma = np.array([
            [d(1).ln()],
            [d(2).ln()],
            [d(3).ln()]])
        result = self.model._recompute_initial_states(log_gamma)
        expected_result = np.array(
            [d(1), d(2), d(3)])
        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_time(self):
        for i in xrange(0, 8):
            log_gamma = np.array([[d(1).ln()]*(8**i)]*2)
            result = self.model._recompute_initial_states(log_gamma)
            expected_result = np.array([d(1)]*2)
            np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_number_of_states(self):
        for i in xrange(0, 7):
            log_gamma = np.array([[d(1).ln()]*2]*(8**i))
            result = self.model._recompute_initial_states(log_gamma)
            expected_result = np.array([d(1)]*(8**i))
            np.testing.assert_array_equal(result, expected_result)
