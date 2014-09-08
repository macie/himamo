# -*- coding: utf-8 -*-
"""
Unit tests for compute logbeta.

"""
from decimal import Decimal as d
import unittest

import numpy as np

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class ComputeLogBetaTestCase(BaseTestCase):
    @classmethod
    def _expected_beta(cls, N, T, num):
        arr = []
        for i in xrange(0, T):
            arr.insert(0, ((d(N)**d(i))*(num**d(i))).ln())
        return np.array([arr]*N)

    def smoke_test_compute_logbeta(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        result = self.model._compute_logbeta(log_a, log_b)
        expected_result = np.array([
            [d(4).ln(), d(2).ln(), d(1).ln()],
            [d(4).ln(), d(2).ln(), d(1).ln()]])
        np.testing.assert_array_equal(result, expected_result)

        log_pi, log_a, log_b = self._testing_parameters_generator(
            3, 2, d(1).ln(), d(2).ln(), d(2).ln())
        result = self.model._compute_logbeta(log_a, log_b)
        expected_result = np.array([
            [d(12).ln(), d(1).ln()],
            [d(12).ln(), d(1).ln()],
            [d(12).ln(), d(1).ln()]])
        np.testing.assert_array_equal(result, expected_result)

    def test_empty_transtition_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(5, 3)
        log_a = np.empty((0, 0))
        with self.assertRaises(ValueError):
            self.model._compute_logbeta(log_a, log_b)

    def test_none_transtition_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(5, 3)
        log_a = None
        with self.assertRaises(TypeError):
            self.model._compute_logbeta(log_a, log_b)

    def test_invalid_size_transtition_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(5, 3)
        log_a = np.ones((1, 1, 1))
        with self.assertRaises(ValueError):
            self.model._compute_logbeta(log_a, log_b)

    def test_empty_emission_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(5, 3)
        log_b = np.empty((0, 0))
        with self.assertRaises(ValueError):
            self.model._compute_logbeta(log_a, log_b)

    def test_none_emission_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(5, 3)
        log_b = None
        with self.assertRaises(TypeError):
            self.model._compute_logbeta(log_a, log_b)

    def test_invalid_size_emission_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(5, 3)
        log_b = np.ones((1, 1, 1))
        with self.assertRaises(ValueError):
            self.model._compute_logbeta(log_a, log_b)

    def test_mismatch_size_transition_and_emission_matrix(self):
        log_a = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        log_b = np.array([[d(1).ln()]])
        with self.assertRaises(ValueError):
            self.model._compute_logbeta(log_a, log_b)

    def test_one_state(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(1, 3)
        result = self.model._compute_logbeta(log_a, log_b)
        expected_result = self._expected_beta(1, 3, d(1))
        np.testing.assert_array_equal(result, expected_result)

    def test_one_time(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(3, 1)
        result = self.model._compute_logbeta(log_a, log_b)
        expected_result = self._expected_beta(3, 1, d(1))
        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_time(self):
        for i in xrange(0, 5):
            log_pi, log_a, log_b = self._testing_parameters_generator(2, 8**i)
            result = self.model._compute_logbeta(log_a, log_b)
            expected_result = self._expected_beta(2, 8**i, d(1))
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-(2*i))

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_number_of_states(self):
        for i in xrange(0, 3):
            log_pi, log_a, log_b = self._testing_parameters_generator(8**i, 2)
            result = self.model._compute_logbeta(log_a, log_b)
            expected_result = self._expected_beta(8**i, 2, d(1))
            np.testing.assert_array_equal(result, expected_result)
