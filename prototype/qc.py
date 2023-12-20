#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import detrend

WIN = 8
def simple_segment(X, y):
    """Segment the pipe measurement
    Assumes profile: \_()_()_()__
    
    X - x coordinate of measurement
    y - thickness
    
    Returns (b1, e1), (b2, e2), (b3, e3)
    bi, ei - beginning and end indices of segment i
    """
    X = np.array(X)
    y = np.array(y)

    # clean ends
    mu_y = np.mean(y)
    clean_ind = np.argwhere((X > 0) & (y > 0) & (y < mu_y))
    b = np.min(clean_ind)
    e = np.max(clean_ind)

    # smooth
    sy = np.mean(np.lib.stride_tricks.sliding_window_view(y, WIN),
                axis=1)

    # XXX: switch to filt_peaks()
    # seek peaks
    peaks = np.argwhere(sy > (np.max(sy[b:e]) * 0.9))
    peaks = peaks[(peaks >= b) & (peaks <= e)]
    peak_b = np.min(peaks)
    peak_e = np.max(peaks)

    # seek valleys
    thin = np.argwhere(sy < mu_y)
    thin_e = thin[(thin > peak_e) & (thin < e)]
    e2 = np.min(thin_e) + (WIN//2)
    thin_b = thin[(thin < peak_b) & (thin > b)]
    e1 = np.max(thin_b)

    return ((b, e1), (e1 + 1, e2), (e2 + 1, e))

def strip(X, y):
    """Data range where values are nonzero
    """
    X = np.array(X)
    y = np.array(y)

    clean_ind = np.argwhere((X > 0) & (y > 0))
    b = np.min(clean_ind)
    e = np.max(clean_ind)
    return (b, e)

def DPT(y, N=3):
    """Discrete Pulse Transform filter

    Apply the L and U filter operators
    with increasing window size   y = L(U(y))
    Removes pulses with width 1, 2, ... up to N

    Input array y must be a numpy array
    """
    for n in range(1, N):
        yprim = operator_U(y, n)  # inner
        yprim = operator_L(yprim, n)  # outer
        y = yprim
    return y

def operator_U(y, n):
    """U (upper) filter operator for an 1d array
    preserves the shape of the data
    """
    ypad = np.pad(y, (0, n), mode="edge")
    ymax = np.max(np.lib.stride_tricks.sliding_window_view(ypad, n + 1),
                  axis=1)
    ypad = np.pad(ymax, (n, 0), mode="edge")
    return np.min(np.lib.stride_tricks.sliding_window_view(ypad, n + 1),
                  axis=1)

def operator_L(y, n):
    """L (lower) filter operator for an 1d array
    preserves the shape of the data
    """
    ypad = np.pad(y, (0, n), mode="edge")
    ymin = np.min(np.lib.stride_tricks.sliding_window_view(ypad, n + 1),
                  axis=1)
    ypad = np.pad(ymin, (n, 0), mode="edge")
    return np.max(np.lib.stride_tricks.sliding_window_view(ypad, n + 1),
                  axis=1)

def find_peaks(x, scale=None):
    """Find peaks in quasi-periodic noisy signals using AMPD algorithm.
    Automatic Multi-Scale Peak Detection originally proposed in
    "An Efficient Algorithm for Automatic Peak Detection in
    Noisy Periodic and Quasi-Periodic Signals", Algorithms 2012, 5, 588-603
    https://doi.org/10.1109/ICRERA.2016.7884365
    Optimized implementation by Igor Gotlibovych, 2018

    https://github.com/ig248/pyampd

    Parameters
    ----------
    x : ndarray
        1-D array on which to find peaks
    scale : int, optional
        specify maximum scale window size of (2 * scale + 1)
    Returns
    -------
    pks: ndarray
        The ordered array of peak indices found in `x`
    """
    x = detrend(x)
    N = len(x)
    L = N // 2
    if scale:
        L = min(scale, L)

    # create LSM matix
    LSM = np.zeros((L, N), dtype=bool)
    for k in np.arange(1, L):
        LSM[k - 1, k:N - k] = (
            (x[0:N - 2 * k] < x[k:N - k]) & (x[k:N - k] > x[2 * k:N])
        )

    # Find scale with most maxima
    G = LSM.sum(axis=1)
    l_scale = np.argmax(G)

    # find peaks that persist on all scales up to l
    pks_logical = np.min(LSM[0:l_scale, :], axis=0)
    pks = np.flatnonzero(pks_logical)
    return pks

def filt_peaks(y, filt=None, scale=None):
    """Filtered peak detector

    1. find peaks using the AMPD algorithm
    2. keep peaks that are within filt*sigma of mean of peaks

    A classical filter value is 3*sigma (filt=3) 
    """
    peaks = find_peaks(y, scale=scale)
    if filt is not None:
        mu = np.mean(y[peaks])
        sigma = np.std(y[peaks])
        delta = np.abs(y[peaks] - mu)
        filtered = peaks[delta < (sigma * filt)]
        peaks = filtered
    return peaks
