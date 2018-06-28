# field_correlation
This code is for calculating field correlations and their significance. The significant test applys for non IID (independent and identically distributed) time series, considering autocorrelations and multiple test problem.

The false discovery rate (FDR) method is applied to take care of the multiple test problem. The FDR code is from Dr. Chris Paciorek (https://www.stat.berkeley.edu/~paciorek/research/code/code.html).

Required packages: numpy, scipy, rpy2, statsmodels, sklearn, collections, pandas

The code includes three files: corr_2d_ttest.py, corr_sig.py and fdr.R.

corr_2d_ttest.ipynb is a Jupyter notebook file as an example showing how to use these functions.
