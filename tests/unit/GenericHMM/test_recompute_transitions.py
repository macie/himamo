# -*- coding: utf-8 -*-
"""
Unit tests for recompute transition matrix.

"""
from decimal import Decimal as d
import unittest

import numpy as np

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class RecomputeTransitionsTestCase(BaseTestCase):
    def smoke_test_recompute_transitions(self):
        # N = 2, T = 3
        log_gamma = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()],
            [d(4).ln(), d(5).ln(), d(6).ln()]])
        log_eta = np.array([
            [
                [d(1).ln(), d(2).ln(), d(3).ln()],
                [d(4).ln(), d(5).ln(), d(6).ln()]],
            [
                [d(2).ln(), d(4).ln(), d(6).ln()],
                [d(18).ln(), d(10).ln(), d(12).ln()]]])
        result = self.model._recompute_transitions(log_gamma, log_eta)
        expected_result = np.array([
            [d(3)/d(3), d(9)/d(3)],
            [d(6)/d(9), d(28)/d(9)]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

        # N = 3, T = 2
        log_gamma = np.array([
            [d(1).ln(), d(2).ln()],
            [d(2).ln(), d(2).ln()],
            [d(3).ln(), d(2).ln()]])
        log_eta = np.array([
            [
                [d(2).ln(), d(1).ln()],
                [d(3).ln(), d(1).ln()],
                [d(4).ln(), d(1).ln()]],
            [
                [d(5).ln(), d(1).ln()],
                [d(6).ln(), d(1).ln()],
                [d(7).ln(), d(1).ln()]],
            [
                [d(8).ln(), d(1).ln()],
                [d(9).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]]])
        result = self.model._recompute_transitions(log_gamma, log_eta)
        expected_result = np.array([
            [d(2)/d(1), d(3)/d(1), d(4)/d(1)],
            [d(5)/d(2), d(6)/d(2), d(7)/d(2)],
            [d(8)/d(3), d(9)/d(3), d(1)/d(3)]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-2)

    def test_empty_loggamma(self):
        log_gamma = np.empty((0, 0))
        log_eta = np.array([
            [
                [d(1).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]],
            [
                [d(1).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]]])
        with self.assertRaises(ValueError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_none_loggamma(self):
        log_gamma = None
        log_eta = np.array([
            [
                [d(1).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]],
            [
                [d(1).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]]])
        with self.assertRaises(TypeError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_invalid_size_loggamma(self):
        log_gamma = np.ones((1, 1, 1))
        log_eta = np.array([
            [
                [d(1).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]],
            [
                [d(1).ln(), d(1).ln()],
                [d(1).ln(), d(1).ln()]]])
        with self.assertRaises(ValueError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_empty_logeta(self):
        log_gamma = np.array([
            [d(1).ln(), d(1).ln()],
            [d(1).ln(), d(1).ln()]])
        log_eta = np.empty((0, 0, 0))
        with self.assertRaises(ValueError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_none_logeta(self):
        log_gamma = np.array([
            [d(1).ln(), d(1).ln()],
            [d(1).ln(), d(1).ln()]])
        log_eta = None
        with self.assertRaises(TypeError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_invalid_size_logeta(self):
        log_gamma = np.array([
            [d(1).ln(), d(1).ln()],
            [d(1).ln(), d(1).ln()]])
        log_eta = np.ones((1, 1))
        with self.assertRaises(ValueError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_mismatch_size_log_gamma_and_log_eta(self):
        log_gamma = np.array([[d(1)]*3]*2, dtype=object)
        log_eta = np.array([[[d(1)]*2]*3]*3, dtype=object)
        with self.assertRaises(ValueError):
            self.model._recompute_transitions(log_gamma, log_eta)

    def test_one_state(self):
        log_gamma = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()]])
        log_eta = np.array([
            [
                [d(1).ln(), d(1).ln(), d(1).ln()]]])
        result = self.model._recompute_transitions(log_gamma, log_eta)
        expected_result = np.array([
            [d(2)/d(3)]])
        np.testing.assert_almost_equal(
            result, expected_result, decimal=self.num_precision-1)

    def test_one_time(self):
        log_gamma = np.array([
            [d(1).ln()],
            [d(2).ln()],
            [d(3).ln()]])
        log_eta = np.array([
            [
                [d(1).ln()],
                [d(1).ln()],
                [d(1).ln()]],
            [
                [d(1).ln()],
                [d(1).ln()],
                [d(1).ln()]],
            [
                [d(1).ln()],
                [d(1).ln()],
                [d(1).ln()]]])
        result = self.model._recompute_transitions(log_gamma, log_eta)
        # we don't know transitions, because we don't have history (time = 1)
        expected_result = np.array([
            [d(0), d(0), d(0)],
            [d(0), d(0), d(0)],
            [d(0), d(0), d(0)]])
        np.testing.assert_array_equal(result, expected_result)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_time(self):
        for i in xrange(1, 5):
            log_gamma = np.array([[d(2).ln()]*(8**i)]*2)
            log_eta = np.array([[[d(1).ln()]*(8**i)]*2]*2)
            result = self.model._recompute_transitions(log_gamma, log_eta)
            expected_result = np.array([[d(1)/d(2)]*2]*2)
            np.testing.assert_almost_equal(
                result, expected_result, decimal=self.num_precision-1)

    @unittest.skip('long duration')
    def test_numerical_stability_if_increased_number_of_states(self):
        for i in xrange(0, 4):
            log_gamma = np.array([[d(2).ln()]*2]*(6**i))
            log_eta = np.array([[[d(1).ln()]*2]*(6**i)]*(6**i))
            result = self.model._recompute_transitions(log_gamma, log_eta)
            expected_result = np.array([[d(1)/d(2)]*(6**i)]*(6**i))
            np.testing.assert_array_equal(result, expected_result)
