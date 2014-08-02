# -*- coding: utf-8 -*-
"""
Functional tests for GenericHMM class.

"""
from decimal import Decimal as d
from decimal import getcontext
import unittest

import mock
import numpy as np

from himamo import GenericHMM


class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM()
        cls.num_precision = getcontext().prec

    @classmethod
    def _testing_parameters_generator(
            cls, N, T, pi_val=d(1), a_val=d(1), b_val=d(1)):
        initial_states = np.array([pi_val]*N, dtype=object)
        transition_matrix = np.array([[a_val]*N]*N, dtype=object)
        observation_symbol = np.array([[b_val]*N]*T, dtype=object)

        return initial_states, transition_matrix, observation_symbol


class BasicMethodsTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM()

    def test_extended_exponent(self):
        result = self.model._eexp(d(1))
        expected_result = d(1).exp()
        self.assertEqual(result, expected_result)

        result = self.model._eexp(d('NaN'))
        expected_result = d(0)
        self.assertEqual(result, expected_result)

    def test_extended_logarithm(self):
        result = self.model._eln(d(1))
        expected_result = d(1).ln()
        self.assertEqual(result, expected_result)

    def test_extended_logarithm_sum(self):
        eln = self.model._eln
        expected_result = eln(d(8))

        result = self.model._elnsum(eln(d(6)), eln(d(2)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(eln(d(2)), eln(d(6)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(eln(d(4)), eln(d(4)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(d('NaN'), eln(d(8)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(eln(d(8)), d('NaN'))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(d('NaN'), d('NaN'))
        self.assertTrue(result.is_nan())

    def test_extended_log_product(self):
        eln = self.model._eln
        expected_result = eln(d(6))

        result = self.model._elnproduct(eln(d(2)), eln(d(3)))
        self.assertEqual(result, expected_result)

        result = self.model._elnproduct(d('NaN'), eln(d(1)))
        self.assertTrue(result.is_nan())

        result = self.model._elnproduct(eln(d(1)), d('NaN'))
        self.assertTrue(result.is_nan())

        result = self.model._elnproduct(d('NaN'), d('NaN'))
        self.assertTrue(result.is_nan())


class LogAlphaTestCase(BaseTestCase):
    @classmethod
    def _expected_alpha(cls, N, T, num):
        arr = []
        for t in xrange(1, T+1):
            arr.append([((d(N)**d(t-1))*(num**d(t))).ln()]*N)
        return np.array(arr)

    def test_small_model_ones(self):
        pi, a, b = self._testing_parameters_generator(5, 3)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logalpha()

        expected_result = self._expected_alpha(5, 3, d(1))

        np.testing.assert_array_equal(result, expected_result)

    def test_small_model_very_small_elements(self):
        very_small_num = d('1e-15')
        pi, a, b = self._testing_parameters_generator(
            5, 3, pi_val=very_small_num, a_val=very_small_num)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logalpha()

        expected_result = self._expected_alpha(5, 3, very_small_num)

        np.testing.assert_array_equal(result, expected_result)

    def test_small_model_very_large_elements(self):
        very_large_num = d('1e15')
        pi, a, b = self._testing_parameters_generator(
            5, 3, pi_val=very_large_num, a_val=very_large_num)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logalpha()

        expected_result = self._expected_alpha(5, 3, very_large_num)

        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_small_model_large_time_ones(self):
        pi, a, b = self._testing_parameters_generator(3, 1000)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logalpha()

        expected_result = self._expected_alpha(3, 1000, d(1))

        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-6)

    @unittest.skip('long duration')
    def test_large_model_ones(self):
        pi, a, b = self._testing_parameters_generator(50, 3)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logalpha()

        expected_result = self._expected_alpha(50, 3, d(1))

        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-2)


class LogBetaTestCase(BaseTestCase):
    @classmethod
    def _expected_beta(cls, N, T, num):
        arr = []
        for i in xrange(0, T):
            arr.insert(0, [((d(N)**d(i))*(num**d(i))).ln()]*N)
        return np.array(arr)

    def test_small_model_ones(self):
        pi, a, b = self._testing_parameters_generator(5, 3)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logbeta()

        expected_result = self._expected_beta(5, 3, d(1))

        np.testing.assert_array_equal(result, expected_result)

    def test_small_model_very_small_elements(self):
        very_small_num = d('1e-15')
        pi, a, b = self._testing_parameters_generator(
            5, 3, pi_val=very_small_num, a_val=very_small_num)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logbeta()

        expected_result = self._expected_beta(5, 3, very_small_num)

        np.testing.assert_array_equal(result, expected_result)

    def test_small_model_very_large_elements(self):
        very_large_num = d('1e15')
        pi, a, b = self._testing_parameters_generator(
            5, 3, pi_val=very_large_num, a_val=very_large_num)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logbeta()

        expected_result = self._expected_beta(5, 3, very_large_num)

        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_small_model_large_time_ones(self):
        pi, a, b = self._testing_parameters_generator(3, 1000)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logbeta()

        expected_result = self._expected_beta(3, 1000, d(1))

        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-6)

    @unittest.skip('long duration')
    def test_large_model_ones(self):
        pi, a, b = self._testing_parameters_generator(50, 3)
        self.model.initial_states = pi
        self.model.transition_matrix = a
        self.model.observation_symbol = b

        result = self.model._compute_logbeta()

        expected_result = self._expected_beta(50, 3, d(1))

        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-2)


class LogGammaTestCase(BaseTestCase):
    @classmethod
    def _expected_gamma(cls, N, T):
        arr = []
        for i in xrange(0, T):
            arr.append([(d(1)/d(N)).ln()]*N)
        return np.array(arr)

    def test_small_model_ones(self):
        self.model._log_alpha = np.array([[d(1)]*5]*3, dtype=object)
        self.model._log_beta = np.array([[d(1)]*5]*3, dtype=object)

        result = self.model._compute_loggamma()

        expected_result = self._expected_gamma(5, 3)

        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

    def test_small_model_very_small_elements(self):
        very_small_num = d('1e-15')
        self.model._log_alpha = np.array([[very_small_num]*5]*3, dtype=object)
        self.model._log_beta = np.array([[very_small_num]*5]*3, dtype=object)

        result = self.model._compute_loggamma()

        expected_result = self._expected_gamma(5, 3)

        np.testing.assert_array_equal(result, expected_result)

    def test_small_model_very_large_elements(self):
        very_large_num = d('1e15')
        self.model._log_alpha = np.array([[very_large_num]*5]*3, dtype=object)
        self.model._log_beta = np.array([[very_large_num]*5]*3, dtype=object)

        result = self.model._compute_loggamma()

        expected_result = self._expected_gamma(5, 3)

        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-15)

    @unittest.skip('long duration')
    def test_small_model_large_time_ones(self):
        self.model._log_alpha = np.array([[d(1)]*3]*1000, dtype=object)
        self.model._log_beta = np.array([[d(1)]*3]*1000, dtype=object)

        result = self.model._compute_loggamma()

        expected_result = self._expected_gamma(3, 1000)

        np.testing.assert_almost_equal(
            result, expected_result)

    @unittest.skip('long duration')
    def test_very_large_model_ones(self):
        self.model._log_alpha = np.array([[d(1)]*1000]*3, dtype=object)
        self.model._log_beta = np.array([[d(1)]*1000]*3, dtype=object)

        result = self.model._compute_loggamma()

        expected_result = self._expected_gamma(1000, 3)

        np.testing.assert_almost_equal(result, expected_result)
