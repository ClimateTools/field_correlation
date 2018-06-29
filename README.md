# Correlations between fields exhibiting long-range dependencies

It's rare to read a climate science paper that doesn't use correlations -- I would wager this holds true in other fields. The trouble is that most of the time, their significance is judged via parametric tests whose assumptions are often violated. [Hu et al. (2017)](https://www.sciencedirect.com/science/article/pii/S0012821X16306823) illustrate how failing to address these concerns can lead to erroneous interpretations.

The code linked herein is for calculating field correlations and their significance. We use [Ebisuzaki's](goo.gl/rr254V), which applies to time series that exhibit non-white spectra, hence violate the independence assumption that is fundamental to the classical t-test (one so classical that all canned routines, from Excel to Matlab, apply it without ever asking you if it applies to your data).  Our method considers both serial dependencies and the test multiplicity problem.
The latter is addressed via the false discovery rate (FDR) method, using code from Dr. Chris Paciorek (https://www.stat.berkeley.edu/~paciorek/research/code/code.html).

We consider two applications:
(1) correlation of field1(time,lat,lon) vs. field2(time,lat,lon)
(2) correlation of field1(time,lat,lon) vs. field2(time)

The code includes three files: corr_2d_ttest.py, corr_sig.py and fdr.R.
You can see them in action in corr_2d_ttest.ipynb  (a Jupyter notebook), which you can either run yourself, or view at: https://nbviewer.jupyter.org/github/ClimateTools/field_correlation/blob/master/corr_2d_ttest.ipynb.

Required packages: numpy, scipy, rpy2, statsmodels, sklearn, collections, pandas
