"""
Some tools for manipulating metalog expressions of the quantile function
as algebraic objects, specifically applying the differential operator.

The provided QuantileSum class defines a set of possible expressions which is
closed under differentiation.
"""

import functools
import operator
import numpy as np
from typing import NamedTuple, Union, List

# NB: Order of derivative corresponds to list index
derivatives_of_ln_y = [
    lambda y: np.log(y/(1-y)),       # 0th derivative
    lambda y: 1/(y*(1-y)),           # 1st derivative
    lambda y: (2*y-1)/(y*(y-1))**2,  # 2nd derivative
    lambda y: 2 * (-3*y**2+3*y-1) / y**3 / (-1 + y*(3 + y*(-3 + y))),  # 3rd derivative; Horner's method is used in denominator
    # Last element reserved for identity (when ln_y_order == -1)
    lambda y: 1
]

class QuantileTerm(NamedTuple):
    coeff: float
    y_centered: int            # Power of (y - 0.5) coefficent
    ln_y_order: int            # How many times we have differentiated the ln(y/1-y) coefficient
                               # -1 is used to indicate that ln(y/1-y) is not present

    def eval(self, y) -> Union[np.inexact,np.ndarray]:
        # We could accelerate further by putting the `if` statements outside the
        # expression, at the cost of writing separate expressions for each case
        return (self.coeff
                * ((y - 0.5)**self.y_centered if self.y_centered else 1)
                * derivatives_of_ln_y[self.ln_y_order](y)
                )

    def diff(self) -> List["QuantileTerm"]:
        """
        Return between 0 and 2 QuantileTerms, their sum corresponding to the derivative.
        """
        terms = []
        # Differentiate (y - 0.5)^n
        if self.y_centered == 1:
            terms += [QuantileTerm(self.coeff, 0, self.ln_y_order)]
        elif self.y_centered > 1:
            # Lower power, then subtract 1 from power
            terms += [QuantileTerm(self.coeff*self.y_centered, self.y_centered-1, self.ln_y_order)]
        # Differentiate coefficient descending from ln_y
        if self.ln_y_order > -1:
            terms += [QuantileTerm(self.coeff, self.y_centered, self.ln_y_order+1)]

        return terms

# Eval transforms for unbounded, semibounded and bounded. Indexed by (lbound > -∞, ubound < ∞)
# - `m` is the evaluation of the sum of quantile terms
# - `y` is the argument that was used to evaluate `m`. In cases where we need to
#   apply the chain rule, this is needed to also evaluate sums of lesser diff order
# - `self` is the `QuantileSum` instance calling this function
def identity(self, y, mp): return mp
def not_implemented(self, y, mp): raise NotImplementedError
def bounded_t_diff0(self, y, m): expm=np.exp(m); return (self.bl + self.bu*expm) / (1+expm)
def bounded_t_diff1(self, y, m1):
    expm=np.exp(self.lower_diffs[0].evalsum(y))
    # TODO: Operation order can likely be optimized
    return (m1*self.bu*expm) / (1 + expm) - (self.bl + self.bu*expm) * m1*expm / (1 + expm)**2
bound_transforms = [
    # Transformations for bounded quantile functions
    {(False, False): identity,                                # Unbounded => no transform
     (True, False) : lambda self, y, m: self.bl + np.exp(m),  # Lower bound
     (False, True) : lambda self, y, m: self.bu - np.exp(-m), # Upper bound
     (True, True)  : bounded_t_diff0                      # Upper & lower bound
     },
    # Transformations for bounded quantile functions differentiated once
    {(False, False): identity,
     (True, False) : lambda self, y, m1: m1*np.exp( self.lower_diffs[0].evalsum(y)),
     (False, True) : lambda self, y, m1: m1*np.exp(-self.lower_diffs[0].evalsum(y)),
     (True, True)  : bounded_t_diff1
     },
    # Derivatives of order 2 and higher are not currently implemented for bounded metalog
    {(False, False): identity,
     (True, False) : not_implemented,
     (False, True) : not_implemented,
     (True, True)  : not_implemented,
     },
    # Transformations for bounded quantile functions differentiated thrice
    {(False, False): identity,
     (True, False) : not_implemented,
     (False, True) : not_implemented,
     (True, True)  : not_implemented,
     },
]

# TODO: Precompute (and cache?) the coefficents when evaluating a sum, since
#       they reappear in multiple terms
class QuantileSum(list):
    """
    List of `QuantileTerm` instances

    Provides `eval` and `diff` methods.
    """
    bl: float   # lower bound
    bu: float   # upper bound
    diffn: int  # number of times expression has been differentiated
    lower_diffs: list  # Store the less-differentiated sums. So for the base
                       # quantile expression it is an empty list, for the once
                       # differentiated sum it is contains the base expression
                       # at position 0, etc. (Position = diff order)

    def __init__(self, iterable=(), lbound=-np.inf, ubound=np.inf, diffn=0, lower_diffs=()):
        lbound = -np.inf if lbound is None else lbound
        ubound =  np.inf if ubound is None else ubound
        assert lbound < ubound, f"Lower bound must be less than upper bound. Received {a}, {b}."
        self.bl, self.bu, self.diffn, self.lower_diffs = lbound, ubound, diffn, list(lower_diffs)
        super().__init__(iterable)

    def __call__(self, y):
        return self.eval(y)

    def evalsum(self, y) -> Union[np.inexact,np.ndarray]:
        return sum(term.eval(y) for term in self)

    def eval(self, y) -> Union[np.inexact,np.ndarray]:
        return bound_transforms[self.diffn][self.bl>-np.inf, self.bu<np.inf](
            self, y, self.evalsum(y))

    def diff(self) -> "QuantileSum":
        # reduce w/ iconcat is the most performant according to https://stackoverflow.com/a/45323085
        return functools.reduce(operator.iconcat, (t.diff() for t in self),
                                QuantileSum(lbound=self.bl, ubound=self.bu,
                                            diffn=self.diffn+1,
                                            lower_diffs=self.lower_diffs+[self]))

    @staticmethod
    def from_metalogistic(metalog: "MetaLogistic") -> "QuantileSum":
        "Create the QuantileSum corresponding the the quantile function of `metalog`"
        avec = metalog.a_vector
        # NB: In Keelin 2016 the a vector is 1-based, while here it is 0-based
        mlsum = QuantileSum(
            (QuantileTerm(avec[0], 0, -1),
             QuantileTerm(avec[1], 0, 0)),
            lbound=metalog.lbound,
            ubound=metalog.ubound
        )
        if len(avec) > 2:
            mlsum.append(QuantileTerm(avec[2], 1, 0))
        if len(avec) > 3:
            mlsum.append(QuantileTerm(avec[3], 1, -1))
            # terms ≥ 4 follow a regular pattern
            y_centered = 1
            if len(avec) % 2:
                # Even number: append None so we don't miss terms
                avec.append(None)
            for aodd, aeven in zip(avec[4::2], avec[5::2]):
                y_centered += 1
                mlsum.append(QuantileTerm(aodd, y_centered, -1))
                if aeven is not None:
                    mlsum.append(QuantileTerm(aeven, y_centered, 0))

        return mlsum

