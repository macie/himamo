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
            cls, N, T, log_pi_val=d(1).ln(),
            log_a_val=d(1).ln(), log_b_val=d(1).ln()):
        initial_states = np.array([log_pi_val]*N, dtype=object)
        transition_matrix = np.array([[log_a_val]*N]*N, dtype=object)
        emission_matrix = np.array([[log_b_val]*T]*N, dtype=object)

        return initial_states, transition_matrix, emission_matrix
