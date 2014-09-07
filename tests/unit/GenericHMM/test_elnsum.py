# -*- coding: utf-8 -*-
"""
Unit tests for extended logarithm sum.

"""
from decimal import Decimal as d
from decimal import Overflow, InvalidOperation

from himamo import GenericHMM
from tests.helpers import BaseTestCase


class ExtendedLogSumTestCase(BaseTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM([1], ['a'])

    def smoke_test_extended_logarithm_sum(self):
        x, y = d(1), d(2)
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

        x, y = d(0), d(1)
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

        x, y = d(0), d('NaN')
        result = self.model._elnsum(x, y)
        expected_result = d(0)
        self.assertEqual(result, expected_result)

        x, y = d('NaN'), d('NaN')
        result = self.model._elnsum(x, y)
        self.assertTrue(result.is_nan())

        x, y = d('NaN'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = d(0)
        self.assertEqual(result, expected_result)

        x, y = d(-1), d(0)
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

        x, y = d(-2), d(-1)
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_empty(self):
        with self.assertRaises(TypeError):
            self.model._elnsum()

    def test_x_is_none(self):
        x, y = None, d(1)
        with self.assertRaises(TypeError):
            self.model._elnsum(x, y)

    def test_y_is_none(self):
        x, y = d(1), None
        with self.assertRaises(TypeError):
            self.model._elnsum(x, y)

    def test_x_and_y_are_none(self):
        x, y = None, None
        with self.assertRaises(TypeError):
            self.model._elnsum(x, y)

    def test_x_is_not_decimal(self):
        x, y = 'some x', d(1)
        with self.assertRaises(TypeError):
            self.model._elnsum(x, y)

    def test_y_is_not_decimal(self):
        x, y = d(1), 'some y'
        with self.assertRaises(TypeError):
            self.model._elnsum(x, y)

    def test_x_and_y_are_not_decimal(self):
        x, y = 'some x', 'some y'
        with self.assertRaises(TypeError):
            self.model._elnsum(x, y)

    def test_very_small_positive_x(self):
        x, y = d('1e-20'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_small_positive_y(self):
        x, y = d(0), d('1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_small_positive_x_and_y(self):
        x, y = d('1e-20'), d('1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_positive_x(self):
        x, y = d('1e20'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_positive_y(self):
        x, y = d(0), d('1e20')
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_positive_x_and_y(self):
        x, y = d('1e-20'), d('1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_plus_infinity_x(self):
        x, y = d('inf'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = d('inf')
        self.assertEqual(result, expected_result)

    def test_plus_infinity_y(self):
        x, y = d(0), d('inf')
        result = self.model._elnsum(x, y)
        expected_result = d('inf')
        self.assertEqual(result, expected_result)

    def test_plus_infinity_x_and_y(self):
        x, y = d('inf'), d('inf')
        with self.assertRaises(InvalidOperation):
            # cannot add INF to -INF
            self.model._elnsum(x, y)

    def test_very_small_negative_x(self):
        x, y = d('-1e-20'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_small_negative_y(self):
        x, y = d(0), d('-1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_small_negative_x_and_y(self):
        x, y = d('-1e-20'), d('-1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_negative_x(self):
        x, y = d('-1e20'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_negative_y(self):
        x, y = d(0), d('-1e20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_negative_x_and_y(self):
        x, y = d('-1e20'), d('-1e20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_minus_infinity_x(self):
        x, y = d('-inf'), d(0)
        result = self.model._elnsum(x, y)
        expected_result = d(0)
        self.assertEqual(result, expected_result)

    def test_minus_infinity_y(self):
        x, y = d(0), d('-inf')
        result = self.model._elnsum(x, y)
        expected_result = d(0)
        self.assertEqual(result, expected_result)

    def test_minus_infinity_x_and_y(self):
        x, y = d('-inf'), d('-inf')
        with self.assertRaises(InvalidOperation):
            # cannot add -INF to INF
            self.model._elnsum(x, y)

    def test_small_positive_x_and_huge_positive_y(self):
        x, y = d('1e-20'), d('1e20')
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_huge_positive_x_and_small_positive_y(self):
        x, y = d('1e20'), d('1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_small_negative_x_and_huge_negative_y(self):
        x, y = d('-1e-20'), d('-1e20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_huge_negative_x_and_small_negative_y(self):
        x, y = d('-1e20'), d('-1e-20')
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_small_positive_x_and_huge_negative_y(self):
        x, y = d('1e-20'), d('-1e20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_huge_positive_x_and_small_negative_y(self):
        x, y = d('1e20'), d('-1e-20')
        result = self.model._elnsum(x, y)
        expected_result = x + (d(1) + (y - x).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_small_negative_x_and_huge_positive_y(self):
        x, y = d('-1e-20'), d('1e20')
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_huge_negative_x_and_small_positive_y(self):
        x, y = d('-1e20'), d('1e-20')
        result = self.model._elnsum(x, y)
        expected_result = y + (d(1) + (x - y).exp()).ln()
        self.assertEqual(result, expected_result)

    def test_minus_infinity_x_and_plus_infinity_y(self):
        x, y = d('-inf'), d('inf')
        result = self.model._elnsum(x, y)
        expected_result = d('inf')
        self.assertEqual(result, expected_result)

    def test_plus_infinity_x_and_minus_infinity_y(self):
        x, y = d('inf'), d('-inf')
        result = self.model._elnsum(x, y)
        expected_result = d('inf')
        self.assertEqual(result, expected_result)
