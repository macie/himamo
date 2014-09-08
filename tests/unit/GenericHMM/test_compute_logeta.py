# -*- coding: utf-8 -*-
"""
Unit tests for compute logeta.

"""
from decimal import Decimal as d
import unittest

import numpy as np

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class ComputeLogEtaTestCase(BaseTestCase):
    @classmethod
    def _expected_logeta(cls, N, T):
        arr = []
        for i in xrange(0, T-1):
            arr.append((d(1)/(d(N)*d(N))).ln())
        arr.append(d(0))
        return np.array([[arr]*N]*N)

    def smoke_test_compute_logeta(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        result = self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)
        expected_result = np.array([
            [
                [(d(1)/d(4)).ln(), (d(1)/d(4)).ln(), d(0)],
                [(d(1)/d(4)).ln(), (d(1)/d(4)).ln(), d(0)]],
            [
                [(d(1)/d(4)).ln(), (d(1)/d(4)).ln(), d(0)],
                [(d(1)/d(4)).ln(), (d(1)/d(4)).ln(), d(0)]]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

        log_pi, log_a, log_b = self._testing_parameters_generator(3, 2)
        log_alpha = np.array([[d(1)]*2]*3, dtype=object)
        log_beta = np.array([[d(1)]*2]*3, dtype=object)
        result = self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)
        expected_result = np.array([
            [
                [(d(1)/d(9)).ln(), d(0)],
                [(d(1)/d(9)).ln(), d(0)],
                [(d(1)/d(9)).ln(), d(0)]],
            [
                [(d(1)/d(9)).ln(), d(0)],
                [(d(1)/d(9)).ln(), d(0)],
                [(d(1)/d(9)).ln(), d(0)]],
            [
                [(d(1)/d(9)).ln(), d(0)],
                [(d(1)/d(9)).ln(), d(0)],
                [(d(1)/d(9)).ln(), d(0)]]])
        np.testing.assert_array_equal(result, expected_result)

    def test_empty_transtition_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        log_a = np.empty((0, 0))
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_none_transtition_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        log_a = None
        with self.assertRaises(TypeError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_invalid_size_transtition_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        log_a = np.ones((1, 1, 1))
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_empty_emission_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        log_b = np.empty((0, 0))
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_none_emission_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        log_b = None
        with self.assertRaises(TypeError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_invalid_size_emission_matrix(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        log_b = np.ones((1, 1, 1))
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_empty_logalpha(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.empty((0, 0), dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_none_logalpha(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = None
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        with self.assertRaises(TypeError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_invalid_size_logalpha(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.ones((1, 1, 1), dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_empty_logbeta(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.empty((0, 0), dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_none_logbeta(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = None
        with self.assertRaises(TypeError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_invalid_size_logbeta(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(2, 3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.ones((1, 1, 1), dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_mismatch_size_transition_and_emission_matrix(self):
        log_a = np.array([[d(1).ln()]*3]*3)
        log_b = np.array([[d(1).ln()]*4])
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_mismatch_size_transition_matrix_and_log_alpha(self):
        log_a = np.array([[d(1).ln()]*3]*3)
        log_b = np.array([[d(1).ln()]*3]*2)
        log_alpha = np.array([[d(1)]*3]*4, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_mismatch_size_transition_matrix_and_log_beta(self):
        log_a = np.array([[d(1).ln()]*3]*3)
        log_b = np.array([[d(1).ln()]*3]*2)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*4, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_mismatch_size_emission_matrix_and_log_alpha(self):
        log_a = np.array([[d(1).ln()]*2]*2)
        log_b = np.array([[d(1).ln()]*3]*3)
        log_alpha = np.array([[d(1)]*3]*4, dtype=object)
        log_beta = np.array([[d(1)]*3]*2, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_mismatch_size_emission_matrix_and_log_beta(self):
        log_a = np.array([[d(1).ln()]*2]*2)
        log_b = np.array([[d(1).ln()]*3]*3)
        log_alpha = np.array([[d(1)]*3]*2, dtype=object)
        log_beta = np.array([[d(1)]*3]*4, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_mismatch_size_log_alpha_and_log_beta(self):
        log_a = np.array([[d(1).ln()]*2]*2)
        log_b = np.array([[d(1).ln()]*3]*2)
        log_alpha = np.array([[d(1)]*3]*3, dtype=object)
        log_beta = np.array([[d(1)]*3]*4, dtype=object)
        with self.assertRaises(ValueError):
            self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)

    def test_one_state(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(1, 3)
        log_alpha = np.array([[d(1)]*3]*1, dtype=object)
        log_beta = np.array([[d(1)]*3]*1, dtype=object)
        result = self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)
        expected_result = self._expected_logeta(1, 3)
        np.testing.assert_array_equal(result, expected_result)

    def test_one_symbol(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(1, 3)
        log_alpha = np.array([[d(1)]*3]*1, dtype=object)
        log_beta = np.array([[d(1)]*3]*1, dtype=object)
        result = self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)
        expected_result = self._expected_logeta(1, 3)
        np.testing.assert_array_equal(result, expected_result)

    def test_one_time(self):
        log_pi, log_a, log_b = self._testing_parameters_generator(3, 1)
        log_alpha = np.array([[d(1)]*1]*3, dtype=object)
        log_beta = np.array([[d(1)]*1]*3, dtype=object)
        result = self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)
        expected_result = self._expected_logeta(3, 1)
        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_time(self):
        for i in xrange(0, 5):
            log_pi, log_a, log_b = self._testing_parameters_generator(2, 8**i)
            log_alpha = np.array([[d(1)]*(8**i)]*2, dtype=object)
            log_beta = np.array([[d(1)]*(8**i)]*2, dtype=object)
            result = self.model._compute_logeta(
                log_a, log_b, log_alpha, log_beta)
            expected_result = self._expected_logeta(2, 8**i)
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-1)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_number_of_states(self):
        for i in xrange(0, 4):
            log_pi, log_a, log_b = self._testing_parameters_generator(8**i, 2)
            log_alpha = np.array([[d(1)]*2]*(8**i), dtype=object)
            log_beta = np.array([[d(1)]*2]*(8**i), dtype=object)
            result = self.model._compute_logeta(log_a, log_b, log_alpha, log_beta)
            expected_result = self._expected_logeta(8**i, 2)
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-(i+1))
