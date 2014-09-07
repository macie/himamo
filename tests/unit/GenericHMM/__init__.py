# -*- coding: utf-8 -*-
"""
Unit tests for GenericHMM class.

"""

from tests.unit.GenericHMM.test_eexp import ExtendedExpTestCase
from tests.unit.GenericHMM.test_eln import ExtendedLogTestCase
from tests.unit.GenericHMM.test_elnsum import ExtendedLogSumTestCase
from tests.unit.GenericHMM.test_elnproduct import ExtendedLogProductTestCase

from tests.unit.GenericHMM.test_compute_logalpha import ComputeLogAlphaTestCase
from tests.unit.GenericHMM.test_compute_logbeta import ComputeLogBetaTestCase
from tests.unit.GenericHMM.test_compute_loggamma import ComputeLogGammaTestCase
from tests.unit.GenericHMM.test_compute_logdelta import ComputeLogDeltaTestCase
from tests.unit.GenericHMM.test_compute_logeta import ComputeLogEtaTestCase

from tests.unit.GenericHMM.test_recompute_initial_states import RecomputeInitialStatesTestCase
from tests.unit.GenericHMM.test_recompute_transitions import RecomputeTransitionsTestCase
from tests.unit.GenericHMM.test_recompute_emissions import RecomputeEmissionsTestCase

__all__ = ['ExtendedExpTestCase', 'ExtendedLogTestCase',
           'ExtendedLogSumTestCase', 'ExtendedLogProductTestCase',
           'ComputeLogAlphaTestCase', 'ComputeLogBetaTestCase',
           'ComputeLogGammaTestCase', 'ComputeLogDeltaTestCase',
           'ComputeLogEtaTestCase', 'RecomputeInitialStatesTestCase',
           'RecomputeTransitionsTestCase', 'RecomputeEmissionsTestCase']
