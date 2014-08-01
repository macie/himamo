# -*- coding: utf-8 -*-
"""
Unit tests for himamo module.

"""
import unittest

import mock

from himamo import GenericHMM


class GenericHMMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM()

    @mock.patch('himamo.himamo.math.exp', side_effect=lambda x: x)
    def test_extended_exponent(self, mocked_log):
        result = self.model._eexp(1)

        expected_result = 1

        self.assertEqual(result, expected_result)

    def test_extended_exponent_NaN(self):
        result = self.model._eexp(None)

        expected_result = 0.0

        self.assertEqual(result, expected_result)

    def test_extended_logarithm_for_zero(self):
        result = self.model._eln(0)

        self.assertIsNone(result)

    @mock.patch('himamo.himamo.math.log', side_effect=lambda x: x)
    def test_extended_logarithm_greater_than_zero(self, mocked_log):
        result = self.model._eln(1)

        expected_result = 1

        self.assertEqual(result, expected_result)

    def test_extended_logarithm_less_than_zero(self):
        with self.assertRaises(ValueError):
            result = self.model._eln(-1)

    def test_extended_log_sum_for_elnx_NaN(self):
        result = self.model._elnsum(None, 1)

        expected_result = 1

        self.assertEqual(result, expected_result)

    def test_extended_log_sum_for_elny_NaN(self):
        result = self.model._elnsum(1, None)

        expected_result = 1

        self.assertEqual(result, expected_result)

    def test_extended_log_sum_for_both_NaN(self):
        result = self.model._elnsum(None, None)

        self.assertIsNone(result)

    @mock.patch('himamo.himamo.GenericHMM._eln', side_effect=lambda x: -x)
    @mock.patch('himamo.himamo.math.exp', side_effect=lambda x: x)
    def test_extended_log_sum_for_elnx_greater_than_elny(
            self, mocked_exp, mocked_eln):
        result = self.model._elnsum(3, 2)

        expected_result = 3

        self.assertEqual(result, expected_result)

    @mock.patch('himamo.himamo.GenericHMM._eln', side_effect=lambda x: -x)
    @mock.patch('himamo.himamo.math.exp', side_effect=lambda x: x)
    def test_extended_log_sum_for_elnx_less_than_elny(
            self, mocked_exp, mocked_eln):
        result = self.model._elnsum(2, 3)

        expected_result = 3

        self.assertEqual(result, expected_result)

    @mock.patch('himamo.himamo.GenericHMM._eln', side_effect=lambda x: -x)
    @mock.patch('himamo.himamo.math.exp', side_effect=lambda x: x)
    def test_extended_log_sum_for_elnx_equal_elny(
            self, mocked_exp, mocked_eln):
        result = self.model._elnsum(3, 3)

        expected_result = 2

        self.assertEqual(result, expected_result)

    def test_extended_log_product(self):
        result = self.model._elnproduct(1, 2)

        expected_result = 3

        self.assertEqual(result, expected_result)

    def test_extended_log_product_for_elnx_NaN(self):
        result = self.model._elnproduct(None, 1)

        self.assertIsNone(result)

    def test_extended_log_product_for_elny_NaN(self):
        result = self.model._elnproduct(1, None)

        self.assertIsNone(result)

    def test_extended_log_product_for_both_NaN(self):
        result = self.model._elnproduct(None, None)

        self.assertIsNone(result)

    def test_compute_logalpha(self):
        pass

    # sum over all states S_i of γ_t (i) should be one, and
    # thus the logarithm of the sum of all states of γ_t (i) should be zero.
    # Similarly, the sum over all state pairs S_i and S_j of ξ_t (i, j)
    # should be one.
