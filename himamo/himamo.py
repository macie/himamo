# -*- coding: utf-8 -*-
"""
Hidden Markov Model python implementation.

"""
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
            x (float or None): A number.

        Returns:
            e**x or None (if x is None).

        """
        if x is None:
            return 0
        else:
            return math.exp(x)

    @classmethod
    def _eln(cls, x):
        """
        Extended logarithm.

        Arguments:
            x (float or None): A number.

        Returns:
            A logarithm x or None (if x is None).

        Reises:
            ValueError if x < 0.

        """
        if x == 0:
            return None
        elif x > 0:
            return math.log(x)
        else:
            raise ValueError

    @classmethod
    def _elnsum(cls, eln_x, eln_y):
        """
        Extended logarithm sum.

        Arguments:
            eln_x (float or None): Natural logarithm from x.
            eln_y (float or None): Natural logarithm from y.

        Returns:
            Sum of logarithms or None (if eln_x and eln_y are None).

        """
        if (eln_x and eln_y) is not None:
            if eln_x > eln_y:
                return eln_x + cls._eln(1 + math.exp(eln_y - eln_x))
            else:
                return eln_y + cls._eln(1 + math.exp(eln_x - eln_y))
        else:
            # returns: eln_y if eln_x is None,
            #          eln_x if eln_y is None,
            #          None if both are None
            return eln_y or eln_x

    @classmethod
    def _elnproduct(cls, eln_x, eln_y):
        """
        Extended logarithm product.

        Arguments:
            eln_x (float or None): Natural logarithm from x.
            eln_y (float or None): Natural logarithm from y.

        Returns:
            Product of logaritms or None (if eln_x and eln_y are None).

        """
        if (eln_x and eln_y) is not None:
            return eln_x + eln_y
        else:
            return None
