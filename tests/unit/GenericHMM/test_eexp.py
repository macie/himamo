# -*- coding: utf-8 -*-
"""
Unit tests for extended exponential function.

"""
from decimal import Decimal as d
from decimal import Overflow
import unittest

import mock

from himamo import GenericHMM


class ExtendedExpTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM([1], ['a'])

    def smoke_test_extended_exponent(self):
        x = d(1)
        result = self.model._eexp(x)
        expected_result = x.exp()
        self.assertEqual(result, expected_result)

        x = d('NaN')
        result = self.model._eexp(x)
        expected_result = d(0)
        self.assertEqual(result, expected_result)

        x = d(-1)
        result = self.model._eexp(x)
        expected_result = x.exp()
        self.assertEqual(result, expected_result)

    def test_empty(self):
        with self.assertRaises(TypeError):
            self.model._eexp()

    def test_x_is_none(self):
        x = None
        with self.assertRaises(TypeError):
            self.model._eexp(x)

    def test_x_is_not_decimal(self):
        x = 'some x'
        with self.assertRaises(TypeError):
            self.model._eexp(x)

    def test_very_small_positive_x(self):
        x = d('1e-20')
        result = self.model._eexp(x)
        expected_result = x.exp()
        self.assertEqual(result, expected_result)

    def test_very_huge_positive_x(self):
        x = d('1e20')
        with self.assertRaises(Overflow):
            self.model._eexp(x)

    def test_plus_infinity_x(self):
        x = d('inf')
        result = self.model._eexp(x)
        expected_result = d('inf')
        self.assertEqual(result, expected_result)

    def test_very_small_negative_x(self):
        x = d('-1e-20')
        result = self.model._eexp(x)
        expected_result = x.exp()
        self.assertEqual(result, expected_result)

    def test_very_huge_negative_x(self):
        x = d('-1e20')
        result = self.model._eexp(x)
        expected_result = x.exp()
        self.assertEqual(result, expected_result)

    def test_minus_infinity_x(self):
        x = d('-inf')
        result = self.model._eexp(x)
        expected_result = d(0)
        self.assertEqual(result, expected_result)
