# -*- coding: utf-8 -*-
"""
Unit tests for extended logarithm function.

"""
from decimal import Decimal as d
from decimal import Overflow
import unittest

import mock

from himamo import GenericHMM


class ExtendedLogTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM([1], ['a'])

    def smoke_test_extended_logarithm(self):
        x = d(1)
        result = self.model._eln(x)
        expected_result = x.ln()
        self.assertEqual(result, expected_result)

        x = d(0)
        result = self.model._eln(x)
        self.assertTrue(result.is_nan())

        x = d(-1)
        with self.assertRaises(ValueError):
            self.model._eln(x)

    def test_empty(self):
        with self.assertRaises(TypeError):
            self.model._eln()

    def test_x_is_none(self):
        x = None
        with self.assertRaises(TypeError):
            self.model._eln(x)

    def test_x_is_not_decimal(self):
        x = 'some x'
        with self.assertRaises(TypeError):
            self.model._eln(x)

    def test_very_small_positive_x(self):
        x = d('1e-20')
        result = self.model._eln(x)
        expected_result = x.ln()
        self.assertEqual(result, expected_result)

    def test_very_huge_positive_x(self):
        x = d('1e20')
        result = self.model._eln(x)
        expected_result = x.ln()
        self.assertEqual(result, expected_result)

    def test_plus_infinity_x(self):
        x = d('inf')
        result = self.model._eln(x)
        expected_result = d('inf')
        self.assertEqual(result, expected_result)

    def test_very_small_negative_x(self):
        x = d('-1e-20')
        with self.assertRaises(ValueError):
            self.model._eln(x)

    def test_very_huge_negative_x(self):
        x = d('-1e20')
        with self.assertRaises(ValueError):
            self.model._eln(x)

    def test_minus_infinity_x(self):
        x = d('-inf')
        with self.assertRaises(ValueError):
            self.model._eln(x)
