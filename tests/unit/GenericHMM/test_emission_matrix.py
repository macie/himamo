# -*- coding: utf-8 -*-
"""
Unit tests for emission matrix property.

"""
from decimal import Decimal as d

import numpy as np

from tests.helpers import BaseTestCase


class EmissionMatrixPropertyTestCase(BaseTestCase):
    def smoke_test_emission_matrix(self):
        emission_matrix = np.array([
            [d(1), d(2)],
            [d(3), d(4)]])
        log_emission_matrix = np.array([
            [d(1).ln(), d(2).ln()],
            [d(3).ln(), d(4).ln()]])

        self.model.emission_matrix = emission_matrix
        result = self.model._log_emission_matrix
        expected_result = log_emission_matrix
        np.testing.assert_array_equal(result, expected_result)

        result = self.model.emission_matrix
        expected_result = emission_matrix
        np.testing.assert_array_equal(result, expected_result)

    def test_empty_emission_matrix(self):
        emission_matrix = np.array([[], []])

        with self.assertRaises(ValueError):
            self.model.emission_matrix = emission_matrix

    def test_none_emission_matrix(self):
        emission_matrix = None

        with self.assertRaises(ValueError):
            self.model.emission_matrix = emission_matrix

    def test_invalid_type_emission_matrix(self):
        emission_matrix = 'invalid value'

        with self.assertRaises(TypeError):
            self.model.emission_matrix = emission_matrix

    def test_invalid_size_emission_matrix(self):
        emission_matrix = np.array(
            [d(1), d(2)])

        with self.assertRaises(ValueError):
            self.model.emission_matrix = emission_matrix

    def test_one_state(self):
        emission_matrix = np.array([
            [d(1), d(2), d(3)]])
        log_emission_matrix = np.array([
            [d(1).ln(), d(2).ln(), d(3).ln()]])

        self.model.emission_matrix = emission_matrix
        result = self.model._log_emission_matrix
        expected_result = log_emission_matrix
        np.testing.assert_array_equal(result, expected_result)

    def test_one_time(self):
        emission_matrix = np.array([
            [d(1)],
            [d(2)],
            [d(3)]])
        log_emission_matrix = np.array([
            [d(1).ln()],
            [d(2).ln()],
            [d(3).ln()]])

        self.model.emission_matrix = emission_matrix
        result = self.model._log_emission_matrix
        expected_result = log_emission_matrix
        np.testing.assert_array_equal(result, expected_result)

    def test_zero_in_emission_matrix(self):
        emission_matrix = np.array([
            [d(0)]])

        self.model.emission_matrix = emission_matrix
        result = self.model._log_emission_matrix[0, 0].is_nan()
        self.assertTrue(result)

    def test_negative_in_emission_matrix(self):
        emission_matrix = np.array([
            [d(-1)]])

        with self.assertRaises(ValueError):
            self.model.emission_matrix = emission_matrix
