import numpy as np
import copy
import xarray as xr
import pandas as pd
from corr_sig import *
from scipy import special
from rpy2.robjects import FloatVector
import rpy2.robjects as robjects
r = robjects.r

def corr_2d_ttest(field1,field2,lat,lon,options,nd):
    '''Calculate correlations over the time dimension for
       field1(time,lat,lon) and field2(time,lat,lon).
       The significant test considers autocorrelation and multiple test problem.
    Input:
        Both field1 and field2 should have the same dimension size, and
        the order of their dimensions should be (time,lat,lon)
        lat, lon are latitude and longitude of field1 or field2
        options: options for corr_sig. Example:
            options = SET(nsim=1000, method='isospectral', alpha=0.05)
        nd: whether field2 is a time series or a 3d array (time,lat,lon)
            nd = 1: field2 is a time series
            nd = 3: field2 is a 3d array
    Output:
        corr: 2d correlations
        latmedian: latitudes of gridcells which do not pass the significant test
        lonmedian: longitudes of gridcells which do not pass the significant test
        latmedian and lonmedian are correspondent with each other.'''

    latt,lont=[],[]
    latna,lonna=[],[]
    nlat=lat.shape[0]
    nlon=lon.shape[0]
    corr = np.zeros((nlat,nlon))
    corr[:]=np.NAN
    pval_med=[]

    for ilat in range(nlat):
        for ilon in range(nlon):
            f1 = field1[:,ilat,ilon]
            if nd==1:
                f2 = field2
            elif nd==3:
                f2 = field2[:,ilat,ilon]
            else:
                raise Exception('Error: nd should be 1 or 3')

            if (not(np.all(np.logical_or(np.logical_or(np.isnan(f1),np.isnan(f2)),np.logical_or(np.isinf(f1),np.isinf(f2)))))) and (not(np.all(f1==0) or np.all(f2==0) or np.all(f1==1) or np.all(f2==1))):
                f1_new = f1[np.logical_not(np.logical_or(np.logical_or(np.isnan(f1),np.isnan(f2)),np.logical_or(np.isinf(f1),np.isinf(f2))))]
                f2_new = f2[np.logical_not(np.logical_or(np.logical_or(np.isnan(f1),np.isnan(f2)),np.logical_or(np.isinf(f1),np.isinf(f2))))]

                rcorr, signif, pval = corr_sig(f1_new, f2_new, options=options)
                corr[ilat,ilon] = rcorr

                pval_med.append(pval)
                latt.append(lat[ilat])
                lont.append(lon[ilon])

                if np.isnan(pval):
                    latna.append(lat[ilat])
                    lonna.append(lon[ilon])
            else:
                latna.append(lat[ilat])
                lonna.append(lon[ilon])

    #start FDR procedure
    pvalr_med = FloatVector(pval_med)
    r.source("fdr.R")
    sig_med = r.fdr(pvalr_med,method="original")
    latmedian=latt[:]
    lonmedian=lont[:]

    #delete grids which are significant
    if sig_med:
        for isig in sorted(sig_med,reverse=True):
            del latmedian[isig-1]
            del lonmedian[isig-1]

    latmedian.extend(latna)
    lonmedian.extend(lonna)

    return corr,latmedian,lonmedian

