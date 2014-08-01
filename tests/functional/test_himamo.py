# -*- coding: utf-8 -*-
"""
Functional tests for himamo module.

"""
from decimal import Decimal
import unittest

import mock

from himamo import GenericHMM


class GenericHMMTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM()

    def test_extended_exponent(self):
        result = self.model._eexp(Decimal(1))
        expected_result = Decimal(1).exp()
        self.assertEqual(result, expected_result)

        result = self.model._eexp(Decimal('NaN'))
        expected_result = Decimal(0)
        self.assertEqual(result, expected_result)

    def test_extended_logarithm(self):
        result = self.model._eln(Decimal(1))
        expected_result = Decimal(1).ln()
        self.assertEqual(result, expected_result)

    def test_extended_logarithm_sum(self):
        eln = self.model._eln
        expected_result = eln(Decimal(8))

        result = self.model._elnsum(eln(Decimal(6)), eln(Decimal(2)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(eln(Decimal(2)), eln(Decimal(6)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(eln(Decimal(4)), eln(Decimal(4)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(Decimal('NaN'), eln(Decimal(8)))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(eln(Decimal(8)), Decimal('NaN'))
        self.assertEqual(result, expected_result)

        result = self.model._elnsum(Decimal('NaN'), Decimal('NaN'))
        self.assertTrue(result.is_nan())

    def test_extended_log_product(self):
        eln = self.model._eln
        expected_result = eln(Decimal(6))

        result = self.model._elnproduct(eln(Decimal(2)), eln(Decimal(3)))
        self.assertEqual(result, expected_result)

        result = self.model._elnproduct(Decimal('NaN'), eln(Decimal(1)))
        self.assertTrue(result.is_nan())

        result = self.model._elnproduct(eln(Decimal(1)), Decimal('NaN'))
        self.assertTrue(result.is_nan())

        result = self.model._elnproduct(Decimal('NaN'), Decimal('NaN'))
        self.assertTrue(result.is_nan())

    def test_compute_logalpha(self):
        pass

    # sum over all states S_i of γ_t (i) should be one, and
    # thus the logarithm of the sum of all states of γ_t (i) should be zero.
    # Similarly, the sum over all state pairs S_i and S_j of ξ_t (i, j)
    # should be one.
