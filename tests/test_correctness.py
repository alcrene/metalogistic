import numpy as np
from scipy import stats
from metalogistic import MetaLogistic


def test_consistency():
    """
    Confirm correctness of constructed metalog distributions by comparing
    with standard distributions.
    """
    for rv in [stats.norm(), stats.cauchy(),                      # (-∞,∞)
               stats.expon(), stats.gamma(2), stats.lognorm(0.5), # [, ∞)
               stats.weibull_max(1,),                             # (-∞, 0]
               stats.beta(1,1)]:                                  # [0, 1]

        yarr = np.linspace(0.01, 0.99, 100)  # The numerics get dicey at the extreme ends of the tail
        xarr = rv.ppf(yarr)
        lbound, ubound = rv.a, rv.b
        # Despite the OLS solver throwing a numerical error warning, more terms
        # remain beneficial
        metalog = MetaLogistic(yarr, xarr, term=20, ubound=ubound, lbound=lbound)

        xarr = rv.ppf(np.linspace(0.05, 0.95, 30))  # Errors get bigger in the tails
        if rv.dist.name == "cauchy":  # Cauchy is known to be pathological, so not too suprising that numerical errors are larger
            atol, rtol = 1e-3, 1e-2
        else:
            atol, rtol = 1e-6, 1e-4
        assert np.allclose(metalog.cdf(xarr), rv.cdf(xarr), atol=atol, rtol=rtol), f"CDF of {rv.dist.name} is not consistent."
        assert np.allclose(metalog.pdf(xarr), rv.pdf(xarr), atol=atol, rtol=rtol), f"PDF of {rv.dist.name} is not consistent."

