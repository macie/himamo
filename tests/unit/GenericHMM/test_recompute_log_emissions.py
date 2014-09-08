# -*- coding: utf-8 -*-
"""
Unit tests for recompute emission matrix.

"""
from decimal import Decimal as d
import unittest

import numpy as np

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class RecomputeLogEmissionsTestCase(BaseTestCase):
    def smoke_test_recompute_log_emissions(self):
        # N = 2, T = 3
        log_gamma = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()],
            [d(4).ln(), d(5).ln(), d(6).ln()]])
        observed_states = np.array(
            [d(1), d(2), d(2)])
        symbols = np.array(
            [d(1), d(2), d(3)])
        result = self.model._recompute_log_emissions(log_gamma,
                                                     observed_states,
                                                     symbols)
        expected_result = np.array([
            [(d(1)/d(6)).ln(), (d(5)/d(6)).ln(), d('NaN')],
            [(d(4)/d(15)).ln(), (d(11)/d(15)).ln(), d('NaN')]])
        np.testing.assert_almost_equal(
            result[:, -2], expected_result[:, -2],
            decimal=self.num_precision-1)
        self.assertTrue(result[0, -1].is_nan())
        self.assertTrue(result[1, -1].is_nan())

        # N = 3, T = 2
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()],
            [d(5).ln(), d(6).ln()]])
        observed_states = np.array(
            [d(1), d(2)])
        symbols = np.array(
            [d(1), d(2)])
        result = self.model._recompute_log_emissions(log_gamma,
                                                     observed_states,
                                                     symbols)
        expected_result = np.array([
            [(d(1)/d(3)).ln(), (d(2)/d(3)).ln()],
            [(d(3)/d(7)).ln(), (d(4)/d(7)).ln()],
            [(d(5)/d(11)).ln(), (d(6)/d(11)).ln()]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

    def test_empty_loggamma(self):
        log_gamma = np.empty((0, 0))
        observed_states = np.array(
            [d(1), d(2)])
        symbols = np.array(
            [d(1), d(2)])
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_none_loggamma(self):
        log_gamma = None
        observed_states = np.array(
            [d(1), d(2)])
        symbols = np.array(
            [d(1), d(2)])
        with self.assertRaises(TypeError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_invalid_size_loggamma(self):
        log_gamma = np.ones((1, 1, 1))
        observed_states = np.array(
            [d(1), d(2)])
        symbols = np.array(
            [d(1), d(2)])
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_empty_observed_states(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        observed_states = np.empty((0))
        symbols = np.array(
            [d(1), d(2)])
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_none_observed_states(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        observed_states = None
        symbols = np.array(
            [d(1), d(2)])
        with self.assertRaises(TypeError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_invalid_size_observed_states(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        observed_states = np.ones((1, 1, 1))
        symbols = np.array(
            [d(1), d(2)])
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_empty_symbols(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        observed_states = np.array(
            [d(1), d(2)])
        symbols = np.empty((0))
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_none_symbols(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        observed_states = np.array(
            [d(1), d(2)])
        symbols = None
        with self.assertRaises(TypeError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_invalid_size_symbols(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])
        observed_states = np.array(
            [d(1), d(2)])
        symbols = np.ones((1, 1))
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_mismatch_size_log_gamma_and_observed_states(self):
        log_gamma = np.array([[d(1).ln()]*2]*3)
        observed_states = np.array([d(1)]*4)
        symbols = np.array([d(1)]*3)
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_mismatch_size_log_gamma_and_symbols(self):
        log_gamma = np.array([[d(1).ln()]*2]*3)
        observed_states = np.array([d(1)]*3)
        symbols = np.array([d(1)]*4)
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_mismatch_size_observed_states_and_symbols(self):
        log_gamma = np.array([[d(1).ln()]*3]*2)
        observed_states = np.array([d(1)]*4)
        symbols = np.array([d(1)]*5)
        with self.assertRaises(ValueError):
            self.model._recompute_log_emissions(log_gamma,
                                                observed_states,
                                                symbols)

    def test_one_state(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()]])
        observed_states = np.array(
            [d(1), d(3), d(1)])
        symbols = np.array(
            [d(1), d(2), d(3)])
        result = self.model._recompute_log_emissions(log_gamma,
                                                     observed_states,
                                                     symbols)
        expected_result = np.array([
            [(d(4)/d(6)).ln(), d('NaN'), (d(2)/d(6)).ln()]])
        np.testing.assert_almost_equal(
            result[:, 0], expected_result[:, 0], decimal=self.num_precision-1)
        self.assertTrue(result[0, 1].is_nan())
        np.testing.assert_almost_equal(
            result[:, 2], expected_result[:, 2], decimal=self.num_precision-1)

    def test_one_time(self):
        log_gamma = np.array([
            [d(1).ln()],
            [d(2).ln()],
            [d(3).ln()]])
        observed_states = np.array(
            [d(1)])
        symbols = np.array(
            [d(1)])
        result = self.model._recompute_log_emissions(log_gamma,
                                                     observed_states,
                                                     symbols)
        expected_result = np.array([
            [(d(1)/d(1)).ln()],
            [(d(2)/d(2)).ln()],
            [(d(3)/d(3)).ln()]])
        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_time(self):
        for i in xrange(0, 3):
            log_gamma = np.array([[d(2).ln()]*(8**i)]*2)
            observed_states = np.array([d(1)]*(8**i))
            symbols = np.array([d(1)]*(8**i))
            result = self.model._recompute_log_emissions(log_gamma,
                                                         observed_states,
                                                         symbols)
            expected_result = np.array([[d(1).ln()]*(8**i)]*2)
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-1)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_number_of_states(self):
        for i in xrange(0, 5):
            log_gamma = np.array([[d(2).ln()]*2]*(8**i))
            observed_states = np.array([d(1)]*2)
            symbols = np.array([d(1)]*2)
            result = self.model._recompute_log_emissions(log_gamma,
                                                         observed_states,
                                                         symbols)
            expected_result = np.array([[d(1).ln()]*2]*(8**i))
            np.testing.assert_array_equal(result, expected_result)
