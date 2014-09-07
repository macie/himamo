# -*- coding: utf-8 -*-
"""
Helper classes for himamo tests.

"""
from decimal import Decimal as d
from decimal import getcontext
import unittest

import numpy as np

from himamo import GenericHMM


class BaseTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericHMM([1], ['a'])
        cls.num_precision = getcontext().prec

    @classmethod
    def _testing_parameters_generator(
            cls, N, T, pi_val=d(1), a_val=d(1), b_val=d(1)):
        initial_states = np.array([pi_val]*N, dtype=object)
        transition_matrix = np.array([[a_val]*N]*N, dtype=object)
        emission_matrix = np.array([[b_val]*T]*N, dtype=object)

        return initial_states, transition_matrix, emission_matrix
