# -*- coding: utf-8 -*-
"""
Unit tests for GenericHMM class.

"""

from tests.unit.GenericHMM.test_eexp import ExtendedExpTestCase
from tests.unit.GenericHMM.test_eln import ExtendedLogTestCase
from tests.unit.GenericHMM.test_elnsum import ExtendedLogSumTestCase
from tests.unit.GenericHMM.test_elnproduct import ExtendedLogProductTestCase


__all__ = ['ExtendedExpTestCase', 'ExtendedLogTestCase',
           'ExtendedLogSumTestCase', 'ExtendedLogProductTestCase']
