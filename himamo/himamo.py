# -*- coding: utf-8 -*-
"""
Hidden Markov Model python implementation.

"""
import decimal
import math


class GenericHMM(object):
    """
    Generic class for Hidden Markov Models.

    """
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
