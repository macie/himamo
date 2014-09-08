# -*- coding: utf-8 -*-
"""
Unit tests for initial states property.

"""
from decimal import Decimal as d

import numpy as np

from tests.helpers import BaseTestCase


class InitialStatesPropertyTestCase(BaseTestCase):
    def smoke_test_initial_states(self):
        initial_states = np.array(
            [d(1), d(2), d(3)])
        log_initial_states = np.array(
            [d(1).ln(), d(2).ln(), d(3).ln()])

        self.model.initial_states = initial_states
        result = self.model._log_initial_states
        expected_result = log_initial_states
        np.testing.assert_array_equal(result, expected_result)

        result = self.model.initial_states
        expected_result = initial_states
        np.testing.assert_array_equal(result, expected_result)

    def test_empty_initial_states(self):
        initial_states = np.array([])

        with self.assertRaises(ValueError):
            self.model.initial_states = initial_states

    def test_none_initial_states(self):
        initial_states = None

        with self.assertRaises(ValueError):
            self.model.initial_states = initial_states

    def test_invalid_type_initial_states(self):
        initial_states = 'invalid value'

        with self.assertRaises(TypeError):
            self.model.initial_states = initial_states

    def test_invalid_size_initial_states(self):
        initial_states = np.array([
            [d(1), d(2)],
            [d(3), d(4)]])

        with self.assertRaises(ValueError):
            self.model.initial_states = initial_states

    def test_one_state(self):
        initial_states = np.array(
            [d(1)])
        log_initial_states = np.array(
            [d(1).ln()])

        self.model.initial_states = initial_states
        result = self.model._log_initial_states
        expected_result = log_initial_states
        np.testing.assert_array_equal(result, expected_result)

    def test_zero_in_initial_states(self):
        initial_states = np.array(
            [d(0)])

        self.model.initial_states = initial_states
        result = self.model._log_initial_states[0].is_nan()
        self.assertTrue(result)

    def test_negative_in_initial_states(self):
        initial_states = np.array(
            [d(-1)])

        with self.assertRaises(ValueError):
            self.model.initial_states = initial_states
