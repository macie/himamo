# -*- coding: utf-8 -*-
"""
Himamo - well documented, wide tested, numerical stable Hidden Markov Model
implementation in python.

Copyright (c) 2014 Maciej Żok <maciek.zok@gmail.com>
MIT License (http://opensource.org/licenses/MIT)

"""
import decimal

import numpy as np


class GenericHMM(object):
    """
    Generic class for Hidden Markov Models.

    """
    def __init__(self, hidden_states=1):
        self._log_alpha = None
        self._log_beta = None
        self._log_gamma = None
        self._log_eta = None

        self.hidden_states = hidden_states
        self.output_alphabet = None
        self.observed_states = None
        self.initial_states = None
        self.transition_matrix = None
        self.observation_symbol = None

    @classmethod
    def _eexp(cls, x):
        """
        Extended exponential function.

        Arguments:
            x (Decimal): A number.

        Returns:
            e**x or Decimal('NaN') (if x is not a number).

        """
        if x.is_nan():
            return decimal.Decimal(0)
        else:
            return x.exp()

    @classmethod
    def _eln(cls, x):
        """
        Extended logarithm.

        Arguments:
            x (Decimal): A number.

        Returns:
            A logarithm x or Decimal('NaN') (if x is not a number).

        Reises:
            ValueError if x < 0.

        """
        if x == 0:
            return decimal.Decimal('NaN')
        elif x > 0:
            return x.ln()
        else:
            raise ValueError

    @classmethod
    def _elnsum(cls, eln_x, eln_y):
        """
        Extended logarithm sum.

        Arguments:
            eln_x (Decimal): Natural logarithm from x.
            eln_y (Decimal): Natural logarithm from y.

        Returns:
            Sum of logarithms or Decimal('NaN') (if eln_x and eln_y is
            not a number).

        """
        if not (eln_x.is_nan() or eln_y.is_nan()):
            dec_1 = decimal.Decimal(1)
            if eln_x > eln_y:
                return eln_x + cls._eln(dec_1 + (eln_y - eln_x).exp())
            else:
                return eln_y + cls._eln(dec_1 + (eln_x - eln_y).exp())
        elif not eln_x.is_nan():
            return eln_x
        elif not eln_y.is_nan():
            return eln_y
        else:
            return decimal.Decimal('NaN')

    @classmethod
    def _elnproduct(cls, eln_x, eln_y):
        """
        Extended logarithm product.

        Arguments:
            eln_x (Decimal): Natural logarithm from x.
            eln_y (Decimal): Natural logarithm from y.

        Returns:
            Product of logaritms or Decimal('NaN') (if eln_x and eln_y is
            not a number)

        """
        if not (eln_x.is_nan() and eln_y.is_nan()):
            return eln_x + eln_y
        else:
            return decimal.Decimal('NaN')

    def _compute_logalpha(self):
        """
        Compute forward variable alpha_t (i) in log space.

            alpha_t (i) = P(O_1, O_2, ..., O_{t-1}, q_t = S_i|lambda)

            alpha_1 (i) = pi_i b_i (O_1)
            alpha_{t+1} (i) = b_i (O_{t+1}) sum_{i=1}^N alpha_t (i) a_{ij}

        Returns:
            An array with logarithm alpha_t (i) elements.

        """
        initial_states = self.initial_states
        observation_symbol = self.observation_symbol
        transition_matrix = self.transition_matrix
        T = observation_symbol.shape[0]
        N = observation_symbol.shape[1]
        log_alpha = np.empty((T, N), dtype=object)

        if initial_states is None:
            raise ValueError

        for j in xrange(0, N):
            log_alpha[0, j] = self._elnproduct(
                self._eln(initial_states[j]),
                self._eln(observation_symbol[0, j]))

        for t in xrange(1, T):
            for j in xrange(0, N):
                logalpha = decimal.Decimal('NaN')
                for i in xrange(0, N):
                    logalpha = self._elnsum(
                        logalpha,
                        self._elnproduct(log_alpha[t-1, i],
                                         self._eln(transition_matrix[i, j])))
                log_alpha[t, j] = self._elnproduct(
                    logalpha,
                    self._eln(observation_symbol[t, j]))

        self._log_alpha = log_alpha
        return log_alpha

    def _compute_logbeta(self):
        """
        Compute backward variable beta_t (i) in log space.

            beta_t (i) = P(O_{t+1}, O_{t+2}, ..., O_T, q_t = S_i|lambda)

            beta_T (i) = 1
            beta_{t+1} (i) = sum_{j=1}^N a_{ij} b_j (O_{t+1}) beta_{t+1} (j)

        Returns:
            An array with logarithm beta_t (i) elements.

        """
        transition_matrix = self.transition_matrix
        observation_symbol = self.observation_symbol
        T = observation_symbol.shape[0]
        N = observation_symbol.shape[1]

        log_beta = np.empty((T, N), dtype=object)

        for i in xrange(0, N):
            log_beta[T-1, i] = decimal.Decimal(0)

        for t in xrange(T - 2, -1, -1):
            for i in xrange(0, N):
                logbeta = decimal.Decimal('NaN')
                for j in xrange(0, N):
                    logbeta = self._elnsum(
                        logbeta,
                        self._elnproduct(
                            self._eln(transition_matrix[i, j]),
                            self._elnproduct(
                                self._eln(observation_symbol[t+1, j]),
                                log_beta[t+1, j])))
                log_beta[t, i] = logbeta

        self._log_beta = log_beta
        return log_beta
