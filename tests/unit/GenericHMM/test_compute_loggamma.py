# -*- coding: utf-8 -*-
"""
Unit tests for compute loggamma.

"""
from decimal import Decimal as d
import unittest

import numpy as np

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class ComputeLogGammaTestCase(BaseTestCase):
    @classmethod
    def _expected_gamma(cls, N, T):
        arr = []
        for i in xrange(0, T):
            arr.append((d(1)/d(N)).ln())
        return np.array([arr]*N)

    def smoke_test_compute_loggamma(self):
        log_alpha = np.array([[d(1)]*2]*3, dtype=object)
        log_beta = np.array([[d(1)]*2]*3, dtype=object)
        result = self.model._compute_loggamma(log_alpha, log_beta)
        expected_result = np.array([
            [(d(1)/d(3)).ln(), (d(1)/d(3)).ln()],
            [(d(1)/d(3)).ln(), (d(1)/d(3)).ln()],
            [(d(1)/d(3)).ln(), (d(1)/d(3)).ln()]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        result = self.model._compute_loggamma(log_alpha, log_beta)
        expected_result = np.array([
            [(d(1)/d(2)).ln(), (d(1)/d(2)).ln(), (d(1)/d(2)).ln()],
            [(d(1)/d(2)).ln(), (d(1)/d(2)).ln(), (d(1)/d(2)).ln()]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

    def test_empty_log_alpha(self):
        log_alpha = np.empty((5, 3), dtype=object)
        log_beta = np.array([[d(1)]*5]*3, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_none_log_alpha(self):
        log_alpha = None
        log_beta = np.array([[d(1)]*5]*3, dtype=object)
        with self.assertRaises(TypeError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_invalid_size_log_alpha(self):
        log_alpha = np.ones((1, 1, 1), dtype=object)
        log_beta = np.array([[d(1)]*5]*3, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_empty_log_beta(self):
        log_alpha = np.array([[d(1)]*5]*3, dtype=object)
        log_beta = np.empty((5, 3), dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_none_log_beta(self):
        log_alpha = np.array([[d(1)]*5]*3, dtype=object)
        log_beta = None
        with self.assertRaises(TypeError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_invalid_size_log_beta(self):
        log_alpha = np.array([[d(1)]*5]*3, dtype=object)
        log_beta = np.ones((1, 1, 1), dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_mismatch_size_logalpha_and_logbeta(self):
        log_alpha = np.array([[d(1)]*5]*3, dtype=object)
        log_beta = np.array([[d(1)]], dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_loggamma(log_alpha, log_beta)

    def test_one_state(self):
        log_alpha = np.array([[d(1)]*3]*1, dtype=object)
        log_beta = np.array([[d(1)]*3]*1, dtype=object)
        result = self.model._compute_loggamma(log_alpha, log_beta)
        expected_result = self._expected_gamma(1, 3)
        np.testing.assert_array_equal(result, expected_result)

    def test_one_time(self):
        log_alpha = np.array([[d(1)]*1]*3, dtype=object)
        log_beta = np.array([[d(1)]*1]*3, dtype=object)
        result = self.model._compute_loggamma(log_alpha, log_beta)
        expected_result = self._expected_gamma(3, 1)
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_symbols(self):
        for i in xrange(0, 5):
            log_alpha = np.array([[d(1)]*(8**i)]*2, dtype=object)
            log_beta = np.array([[d(1)]*(8**i)]*2, dtype=object)
            result = self.model._compute_loggamma(log_alpha, log_beta)
            expected_result = self._expected_gamma(2, 8**i)
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-1)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_number_of_states(self):
        for i in xrange(0, 5):
            log_alpha = np.array([[d(1)]*2]*(8**i), dtype=object)
            log_beta = np.array([[d(1)]*2]*(8**i), dtype=object)
            result = self.model._compute_loggamma(log_alpha, log_beta)
            expected_result = self._expected_gamma(8**i, 2)
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-((i//2)+1))
