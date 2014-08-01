# -*- coding: utf-8 -*-
"""
Unit tests for himamo module.

"""
from decimal import Decimal
import unittest

import mock

from himamo import GenericHMM


class GenericHMMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM()

    @mock.patch('himamo.himamo.decimal.Decimal.exp', return_value='exp(x)')
    def test_extended_exponent(self, mocked_log):
        result = self.model._eexp(Decimal(1))

        expected_result = 'exp(x)'

        self.assertEqual(result, expected_result)

    def test_extended_exponent_NaN(self):
        result = self.model._eexp(Decimal('NaN'))

        expected_result = Decimal(0)

        self.assertEqual(result, expected_result)

    def test_extended_logarithm_for_zero(self):
        result = self.model._eln(Decimal(0))

        self.assertTrue(result.is_nan())

    @mock.patch('himamo.himamo.decimal.Decimal.ln', return_value='ln(x)')
    def test_extended_logarithm_greater_than_zero(self, mocked_ln):
        result = self.model._eln(Decimal(1))

        expected_result = 'ln(x)'

        self.assertEqual(result, expected_result)

    def test_extended_logarithm_less_than_zero(self):
        with self.assertRaises(ValueError):
            result = self.model._eln(Decimal(-1))

    def test_extended_log_sum_for_elnx_NaN(self):
        result = self.model._elnsum(Decimal('NaN'), Decimal(1))

        expected_result = Decimal(1)

        self.assertEqual(result, expected_result)

    def test_extended_log_sum_for_elny_NaN(self):
        result = self.model._elnsum(Decimal(1), Decimal('NaN'))

        expected_result = Decimal(1)

        self.assertEqual(result, expected_result)

    def test_extended_log_sum_for_both_NaN(self):
        result = self.model._elnsum(Decimal('NaN'), Decimal('NaN'))

        self.assertTrue(result.is_nan())

    @mock.patch('himamo.himamo.GenericHMM._eln', return_value=Decimal(0))
    def test_extended_log_sum_for_elnx_greater_than_elny(self, mocked_eln):
        result = self.model._elnsum(Decimal(3), Decimal(2))

        expected_result = Decimal(3)

        self.assertEqual(result, expected_result)

    @mock.patch('himamo.himamo.GenericHMM._eln', return_value=Decimal(0))
    def test_extended_log_sum_for_elnx_less_than_elny(self, mocked_eln):
        result = self.model._elnsum(Decimal(2), Decimal(3))

        expected_result = Decimal(3)

        self.assertEqual(result, expected_result)

    @mock.patch('himamo.himamo.GenericHMM._eln', return_value=Decimal(0))
    def test_extended_log_sum_for_elnx_equal_elny(self, mocked_eln):
        result = self.model._elnsum(Decimal(3), Decimal(3))

        expected_result = Decimal(3)

        self.assertEqual(result, expected_result)

    def test_extended_log_product(self):
        result = self.model._elnproduct(Decimal(1), Decimal(2))

        expected_result = Decimal(3)

        self.assertEqual(result, expected_result)

    def test_extended_log_product_for_elnx_NaN(self):
        result = self.model._elnproduct(Decimal('NaN'), Decimal(1))

        self.assertTrue(result.is_nan())

    def test_extended_log_product_for_elny_NaN(self):
        result = self.model._elnproduct(Decimal(1), Decimal('NaN'))

        self.assertTrue(result.is_nan())

    def test_extended_log_product_for_both_NaN(self):
        result = self.model._elnproduct(Decimal('NaN'), Decimal('NaN'))

        self.assertTrue(result.is_nan())

    def test_compute_logalpha(self):
        pass

    # sum over all states S_i of γ_t (i) should be one, and
    # thus the logarithm of the sum of all states of γ_t (i) should be zero.
    # Similarly, the sum over all state pairs S_i and S_j of ξ_t (i, j)
    # should be one.
