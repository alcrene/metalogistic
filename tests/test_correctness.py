import numpy as np
from scipy import stats
from metalogistic import MetaLogistic
from metalogistic.algebra import QuantileSum


def test_consistency():
    """
    Confirm correctness of constructed metalog distributions by comparing
    with standard distributions.
    """
    for rv in [stats.norm(), stats.cauchy(),                      # (-∞,∞)
               stats.expon(), stats.gamma(2), stats.lognorm(0.5), # [, ∞)
               stats.weibull_max(1, loc=3.5),                     # (-∞, 3.5]
               stats.beta(1,1)]:                                  # [0, 1]

        yarr = np.linspace(0.01, 0.99, 100)  # The numerics get dicey at the extreme ends of the tail
        xarr = rv.ppf(yarr)
        shape, loc, scale = rv.dist._parse_args(*rv.args, **rv.kwds)  # Sadly, I don't know a better way to get `loc` and `scale`
        lbound = rv.a*scale + loc   # NB: a, b bounds are given for the standardized form
        ubound = rv.b*scale + loc   #     => we need to invert the transform (x - loc)/scale
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

def test_derivatives_lny():
    from metalogistic.algebra import derivatives_of_ln_y as diffs

    yarr = np.linspace(0.1, 0.9, 4000)
    xnarr = diffs[0](yarr)  # 0-th order derivative => original function
    # Check 1st derivative
    xnarr = (xnarr[2:] - xnarr[:-2])/(yarr[2:]-yarr[:-2])  # Evaluate derivative with centered differences
    yarr = yarr[1:-1]
    assert np.allclose(xnarr, diffs[1](yarr))
    # Check 2nd derivative
    xnarr = (xnarr[2:] - xnarr[:-2])/(yarr[2:]-yarr[:-2])
    yarr = yarr[1:-1]
    assert np.allclose(xnarr, diffs[2](yarr))
    # Check 3rd derivative
    xnarr = (xnarr[2:] - xnarr[:-2])/(yarr[2:]-yarr[:-2])
    yarr = yarr[1:-1]
    assert np.allclose(xnarr, diffs[3](yarr), rtol=1e-4)

def test_derivative_quantile():
    """
    """
    for nterms in [3, 6, 20]:  # Test with both even and odd terms, <4 and >4
        for rv in [stats.norm(), #stats.cauchy(),                      # (-∞,∞)
                   stats.expon(), stats.gamma(2), stats.lognorm(0.5), # [, ∞)
                   stats.weibull_max(1, loc=3.5),                     # (-∞, 3.5]
                   stats.beta(1,1)]:                                  # [0, 1]

            yarr = np.linspace(0.01, 0.99, 100)  # The numerics get dicey at the extreme ends of the tail
            xarr = rv.ppf(yarr)
            shape, loc, scale = rv.dist._parse_args(*rv.args, **rv.kwds)  # Sadly, I don't know a better way to get `loc` and `scale`
            lbound = rv.a*scale + loc   # NB: a, b bounds are given for the standardized form
            ubound = rv.b*scale + loc   #     => we need to invert the transform (x - loc)/scale
            metalog = MetaLogistic(yarr, xarr, term=3, ubound=ubound, lbound=lbound)

            mlogsum = QuantileSum.from_metalogistic(metalog)
            assert np.allclose(mlogsum.eval(yarr), metalog.ppf(yarr)), f"QuantileSum not equal to RV's PPF for distribution {rv.dist.name}"

            # Test first derivative
            mldiff1 = mlogsum.diff()
            # Compute diff numerically with centered differences
            yarr = np.linspace(0.3, 0.7, 4000)  # Focus on center, where numerical diff is more precise
            xarr = metalog.ppf(yarr)
            x1arr = (xarr[2:]-xarr[:-2])/(yarr[2:]-yarr[:-2])
            y1arr = yarr[1:-1]
            assert np.allclose(mldiff1.eval(y1arr), x1arr), f"First derivative {rv.dist.name} not equal."

            # Test second derivative
            if lbound > -np.inf or ubound < np.inf:
                # Higher order derivatives are not currently implemented for bounded metalogs
                continue
            mldiff2 = mldiff1.diff()
            # Compute diff numerically with centered differences
            x2arr = (x1arr[2:]-x1arr[:-2])/(y1arr[2:]-y1arr[:-2])
            y2arr = y1arr[1:-1]
            assert np.allclose(mldiff2.eval(y2arr), x2arr), f"Second derivative {rv.dist.name} not equal."


# if __name__ == "__main__":
#     test_consistency()