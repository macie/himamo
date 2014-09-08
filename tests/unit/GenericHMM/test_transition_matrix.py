# -*- coding: utf-8 -*-
"""
Unit tests for transition matrix property.

"""
from decimal import Decimal as d

import numpy as np

from tests.helpers import BaseTestCase


class TransitionMatrixPropertyTestCase(BaseTestCase):
    def smoke_test_transition_matrix(self):
        transition_matrix = np.array([
            [d(1), d(2)],
            [d(3), d(4)]])
        log_transition_matrix = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])

        self.model.transition_matrix = transition_matrix
        result = self.model._log_transition_matrix
        expected_result = log_transition_matrix
        np.testing.assert_array_equal(result, expected_result)

        result = self.model.transition_matrix
        expected_result = transition_matrix
        np.testing.assert_array_equal(result, expected_result)

    def test_empty_transition_matrix(self):
        transition_matrix = np.array([[], []])

        with self.assertRaises(ValueError):
            self.model.transition_matrix = transition_matrix

    def test_none_transition_matrix(self):
        transition_matrix = None

        with self.assertRaises(ValueError):
            self.model.transition_matrix = transition_matrix

    def test_invalid_type_transition_matrix(self):
        transition_matrix = 'invalid value'

        with self.assertRaises(TypeError):
            self.model.transition_matrix = transition_matrix

    def test_invalid_size_transition_matrix(self):
        transition_matrix = np.array([
            [d(1), d(2)]])

        with self.assertRaises(ValueError):
            self.model.transition_matrix = transition_matrix

    def test_one_state(self):
        transition_matrix = np.array([
            [d(1)]])
        log_transition_matrix = np.array([
            [d(1).ln()]])

        self.model.transition_matrix = transition_matrix
        result = self.model._log_transition_matrix
        expected_result = log_transition_matrix
        np.testing.assert_array_equal(result, expected_result)

    def test_zero_in_transition_matrix(self):
        transition_matrix = np.array([
            [d(0)]])

        self.model.transition_matrix = transition_matrix
        result = self.model._log_transition_matrix[0, 0].is_nan()
        self.assertTrue(result)

    def test_negative_in_transition_matrix(self):
        transition_matrix = np.array([
            [d(-1)]])

        with self.assertRaises(ValueError):
            self.model.transition_matrix = transition_matrix
