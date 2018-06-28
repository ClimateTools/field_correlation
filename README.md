# field_correlation
This code is for calculating field correlations and their significance. The significant test applys for non IID (independent and identically distributed) time series, considering autocorrelations and multiple test problem.

The false discovery rate (FDR) method is applied to take care of the multiple test problem. The FDR code is from Dr. Chris Paciorek (https://www.stat.berkeley.edu/~paciorek/research/code/code.html).

The field correlation can be done with this code is:
(1) field1(time,lat,lon) vs. field2(time,lat,lon)
(2) field1(time,lat,lon) vs. field2(time)

Required packages: numpy, scipy, rpy2, statsmodels, sklearn, collections, pandas

The code includes three files: corr_2d_ttest.py, corr_sig.py and fdr.R.

corr_2d_ttest.ipynb is a Jupyter notebook file as an example showing how to use these functions.
