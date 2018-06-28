#!/usr/bin/env python3
"""A Python 3 version of corr_sig.m """

import numpy as np
from scipy.stats import pearsonr
from scipy.stats.mstats import gmean
from collections import namedtuple
from scipy.stats import t as stu
from scipy.stats import gaussian_kde
import statsmodels.api as sm
#  from statsmodels.tsa.stattools import acf
from sklearn import preprocessing

SET = namedtuple("SET", "nsim method alpha")
options = SET(nsim=1000, method='isospectral', alpha=0.05)

debug = False


def corr_sig(x, y, options=options):
    """ Estimates the significance of correlations between non IID time series by 3 independent methods:
    1) 'ttest': T-test where d.o.f are corrected for the effect of serial correlation
    2) 'isopersistent': AR(1) modeling of x and y.
    3) 'isospectral': phase randomization of original inputs. (default)

    The T-test is parametric test, hence cheap but usually wrong except in idyllic circumstances.
    The others are non-parametric, but their computational requirements scales with nsim.

    Args:
        x, y (array): vector of (real) numbers of identical length, no NaNs allowed
        options: structure specifying options [default]:
            - nsim: the number of simulations [1000]
            - method: methods 1-3 above ['isospectral']
            - alpha: significance level for critical value estimation [0.05]

     Returns:
         r (real): correlation between x and y
         signif (boolean): true (1) if significant; false (0) otherwise
         p (real): Fraction of time series with higher correlation coefficents than observed (approximates the p-value).
            Note that signif = True if and only if p <= alpha.

    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    assert np.size(x) == np.size(y), 'The size of X and the size of Y should be the same!'

    if options.method == 'ttest':
        (r, signif, p) = corr_ttest(x, y, alpha=options.alpha)
    elif options.method == 'isopersistent':
        (r, signif, p) = corr_isopersist(x, y, alpha=options.alpha, nsim=options.nsim)
    elif options.method == 'isospectral':
        (r, signif, p) = corr_isospec(x, y, alpha=options.alpha, nsim=options.nsim)

    return r, signif, p


def corr_ttest(x, y, alpha=0.05):
    """ Estimates the significance of correlations between 2 time series using
    the classical T-test with degrees of freedom modified for autocorrelation.

    This function creates 'nsim' random time series that have the same power
    spectrum as the original time series but with random phases.

    Args:
        x, y (array): vectors of (real) numbers with identical length, no NaNs allowed
        alpha (real): significance level for critical value estimation [default: 0.05]

    Returns:
        r (real): correlation between x and y
        signif (boolean): true (1) if significant; false (0) otherwise
        pval (real): test p-value (the probability of the test statstic exceeding the observed one by chance alone)

    """
    r = pearsonr(x, y)[0]

    g1 = ar1_fit(x)
    g2 = ar1_fit(y)

    # Have tested the acf function in statsmodels.tsa.stattools, the result is very close to the return from ARMA fit.
    #  g1_acf = acf(x)[1]
    #  g2_acf = acf(y)[1]

    N = np.size(x)

    Nex = N * (1-g1) / (1+g1)
    Ney = N * (1-g2) / (1+g2)

    Ne = gmean([Nex+Ney])
    assert Ne >= 10, 'Too few effective d.o.f. to apply this method!'

    df = Ne - 2
    t = np.abs(r) * np.sqrt(df/(1-r**2))

    pval = 2 * stu.cdf(-np.abs(t), df)

    signif = pval <= alpha

    if debug:
        print(r)
        print(g1, g2)
        #  print(g1_acf, g2_acf)
        print(Nex, Ney, Ne)
        print(t)

    return r, signif, pval


def corr_isopersist(x, y, alpha=0.05, nsim=1000):
    ''' Computes correlation between two timeseries, and their significance.
    The latter is gauged via a non-parametric (Monte Carlo) simulation of
    correlations with nsim AR(1) processes with identical persistence
    properties as x and y ; the measure of which is the lag-1 autocorrelation (g).

    Args:
        x, y (array): vectors of (real) numbers with identical length, no NaNs allowed
        alpha (real): significance level for critical value estimation [default: 0.05]
        nsim (int): number of simulations [default: 1000]

    Returns:
        r (real): correlation between x and y
        signif (boolean): true (1) if significant; false (0) otherwise
        pval (real): test p-value (the probability of the test statstic exceeding the observed one by chance alone)

    Remarks:
        The probability of obtaining a test statistic at least as extreme as the one actually observed,
        assuming that the null hypothesis is true.

        The test is 1 tailed on |r|: Ho = { |r| = 0 }, Ha = { |r| > 0 }
        The test is rejected (signif = 1) if pval <= alpha, otherwise signif=0;

        (Some Rights Reserved) Hepta Technologies, 2009
        v1.0 USC, Aug 10 2012, based on corr_signif.m

    '''

    r = pearsonr(x, y)[0]
    ra = np.abs(r)

    x_red, g1 = isopersistent_rn(x, nsim)
    y_red, g2 = isopersistent_rn(y, nsim)

    rs = np.zeros(nsim)
    for i in np.arange(nsim):
        rs[i] = pearsonr(x_red[:, i], y_red[:, i])[0]

    rsa = np.abs(rs)

    xi = np.linspace(0, 1.1*np.max([ra, np.max(rsa)]), 200)
    kde = gaussian_kde(rsa)
    prob = kde(xi).T

    diff = np.abs(ra - xi)
    #  min_diff = np.min(diff)
    pos = np.argmin(diff)

    pval = np.trapz(prob[pos:], xi[pos:])

    rcrit = np.percentile(rsa, 100*(1-alpha))
    signif = ra >= rcrit

    return r, signif, pval


def isopersistent_rn(X, M):
    ''' Generates M realization of a red noise [i.e. AR(1)] process
    with same persistence properties as X (Mean and variance are also preserved).

    Args:
        X (array): vector of (real) numbers as a time series, no NaNs allowed
        M (int): number of simulations

    Returns:
        red (matrix): N rows by M columns matrix of an AR1 process
        g (real): lag-1 autocorrelation coefficient

    Remarks:
        (Some Rights Reserved) Hepta Technologies, 2008

    '''
    N = np.size(X)
    mu = np.mean(X)
    sig = np.std(X, ddof=1)

    g = ar1_fit(X)
    red = red_noise(N, M, g)
    m = np.mean(red)
    s = np.std(red, ddof=1)

    red_n = (red - m) / s
    red_z = red_n * sig + mu

    return red_z, g


def ar1_fit(ts):
    ''' Return the lag-1 autocorrelation from ar1 fit.

    Args:
        ts (array): vector of (real) numbers as a time series

    Returns:
        g (real): lag-1 autocorrelation coefficient

    '''
    ar1_mod = sm.tsa.ARMA(ts, (1, 0)).fit()
    g = ar1_mod.params[1]

    return g


def red_noise(N, M, g):
    ''' Produce AR1 process matrix with nv = [N M] and lag-1 autocorrelation g.

    Args:
        N, M (int): dimensions as N rows by M columns
        g (real): lag-1 autocorrelation coefficient

    Returns:
        red (matrix): N rows by M columns matrix of an AR1 process

    Remarks:
        (Some Rights Reserved) Hepta Technologies, 2008
        J.E.G., GaTech, Oct 20th 2008

    '''
    red = np.zeros(shape=(N, M))
    red[0, :] = np.random.randn(1, M)
    for i in np.arange(1, N):
        red[i, :] = g * red[i-1, :] + np.random.randn(1, M)

    #  if debug:
        #  print(red)

    return red


def corr_isospec(x, y, alpha=0.05, nsim=1000):
    ''' Estimates the significance of correlations between non IID
    time series by phase randomization of original inputs.

    This function creates 'nsim' random time series that have the same power
    spectrum as the original time series but random phases.

    Args:
        x, y (array): vectors of (real) numbers with identical length, no NaNs allowed
        alpha (real): significance level for critical value estimation [default: 0.05]
        nsim (int): number of simulations [default: 1000]

    Returns:
        r (real): correlation between x and y
        signif (boolean): true (1) if significant; false (0) otherwise
        F : Fraction of time series with higher correlation coefficents than observed (approximates the p-value).

    References:
        - Ebisuzaki, W, 1997: A method to estimate the statistical
        significance of a correlation when the data are serially correlated.
        J. of Climate, 10, 2147-2153.

        - Prichard, D., Theiler, J. Generating Surrogate Data for Time Series
        with Several Simultaneously Measured Variables (1994)
        Physical Review Letters, Vol 73, Number 7

        (Some Rights Reserved) USC Climate Dynamics Lab, 2012.

    '''
    r = pearsonr(x, y)[0]

    # generate phase-randomized samples using the Theiler & Prichard method
    Xsurr = phaseran(x, nsim)
    Ysurr = phaseran(y, nsim)

    # compute correlations
    Xs = preprocessing.scale(Xsurr)
    Ys = preprocessing.scale(Ysurr)

    n = np.size(x)
    C = np.dot(np.transpose(Xs), Ys) / (n-1)
    rSim = np.diag(C)

    # compute fraction of values higher than observed
    F = np.sum(np.abs(rSim) >= np.abs(r)) / nsim

    # establish significance
    signif = F < alpha  # significant or not?

    return r, signif, F


def phaseran(recblk, nsurr):
    ''' Phaseran by Carlos Gias: http://www.mathworks.nl/matlabcentral/fileexchange/32621-phase-randomization/content/phaseran.m

    Args:
        recblk (2D array): Row: time sample. Column: recording.
            An odd number of time samples (height) is expected.
            If that is not the case, recblock is reduced by 1 sample before the surrogate data is created.
            The class must be double and it must be nonsparse.

        nsurr (int): is the number of image block surrogates that you want to generate.

    Returns:
        surrblk: 3D multidimensional array image block with the surrogate datasets along the third dimension

    Reference:
        Prichard, D., Theiler, J. Generating Surrogate Data for Time Series with Several Simultaneously Measured Variables (1994)
        Physical Review Letters, Vol 73, Number 7

    '''
    # Get parameters
    nfrms = recblk.shape[0]

    if nfrms % 2 == 0:
        nfrms = nfrms-1
        recblk = recblk[0:nfrms]

    len_ser = int((nfrms-1)/2)
    interv1 = np.arange(1, len_ser+1)
    interv2 = np.arange(len_ser+1, nfrms)

    # Fourier transform of the original dataset
    fft_recblk = np.fft.fft(recblk)

    # Creat nsurr timeseries of random numbers (0,1)
    ph_rnd = np.random.rand(len_ser,nsurr)

    # Create the random phases for all the time series
    ph_interv1 = np.exp(2*np.pi*1j*ph_rnd)
    ph_interv2 = np.conj(np.flipud(ph_interv1))

    # Randomize all the time series simultaneously
    fft_recblk_surr = np.tile(fft_recblk[:,None],(1,nsurr))
    fft_recblk_surr[interv1,:] = np.tile(fft_recblk[interv1,None],(1,nsurr)) * ph_interv1
    fft_recblk_surr[interv2,:] = np.tile(fft_recblk[interv2,None],(1,nsurr)) * ph_interv2

    # Inverse transform
    surrblk = np.real(np.fft.ifft(fft_recblk_surr,axis=0))

    return surrblk
