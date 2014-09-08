# -*- coding: utf-8 -*-
"""
Unit tests for GenericHMM class.

"""

from tests.unit.GenericHMM.test_eexp import ExtendedExpTestCase
from tests.unit.GenericHMM.test_eln import ExtendedLogTestCase
from tests.unit.GenericHMM.test_elnsum import ExtendedLogSumTestCase
from tests.unit.GenericHMM.test_elnproduct import ExtendedLogProductTestCase

from tests.unit.GenericHMM.test_initial_states import InitialStatesPropertyTestCase
from tests.unit.GenericHMM.test_transition_matrix import TransitionMatrixPropertyTestCase
from tests.unit.GenericHMM.test_emission_matrix import EmissionMatrixPropertyTestCase

from tests.unit.GenericHMM.test_compute_logalpha import ComputeLogAlphaTestCase
from tests.unit.GenericHMM.test_compute_logbeta import ComputeLogBetaTestCase
from tests.unit.GenericHMM.test_compute_loggamma import ComputeLogGammaTestCase
from tests.unit.GenericHMM.test_compute_logdelta import ComputeLogDeltaTestCase
from tests.unit.GenericHMM.test_compute_logeta import ComputeLogEtaTestCase

from tests.unit.GenericHMM.test_recompute_log_initial_states import RecomputeLogInitialStatesTestCase
from tests.unit.GenericHMM.test_recompute_log_transitions import RecomputeLogTransitionsTestCase
from tests.unit.GenericHMM.test_recompute_log_emissions import RecomputeLogEmissionsTestCase

__all__ = ['ExtendedExpTestCase', 'ExtendedLogTestCase',
           'ExtendedLogSumTestCase', 'ExtendedLogProductTestCase',
           'InitialStatesPropertyTestCase', 'TransitionMatrixPropertyTestCase',
           'EmissionMatrixPropertyTestCase', 'ComputeLogAlphaTestCase',
           'ComputeLogBetaTestCase', 'ComputeLogGammaTestCase',
           'ComputeLogDeltaTestCase', 'ComputeLogEtaTestCase',
           'RecomputeLogInitialStatesTestCase',
           'RecomputeLogTransitionsTestCase',
           'RecomputeLogEmissionsTestCase']
