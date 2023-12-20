#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import qc
from sklearn.feature_selection import f_classif

EPSILON = 0.0001
def deposit(X, te, rf, kernel_shape="flat", kernel_w=20):
    """Mass deposit model
    
    X - carriage position
    te - total extruder throughput
    rf - rotation factor (proportional to rot. speed)
    """

    hist_w = int(np.max(X)) + kernel_w
    D = np.zeros(hist_w)
    if kernel_shape == "flat":
        kernel = np.ones(kernel_w) / kernel_w
    else:
        raise NotImplementedError("kernel", kernel_shape, "not implemented")
    M = te / (rf + EPSILON)
    
    # XXX: vectorize?
    for i in range(M.shape[0]):
        j = int(X[i])
        k = j + kernel_w
        D[j:k] = D[j:k] + (kernel * M[i])

    return D

def interp_deposit(X, te, rf, kernel_shape="flat", kernel_w=20):
    D = deposit(X, te, rf, kernel_shape, kernel_w)
    X_size = int(X.max() + 1)
    Y = np.interp(X, np.arange(X_size), D[:X_size])
    return Y

def _thickness(measurements, dpt_N=3, ampd_Z=3.0,
               mode="min"):
    m_points = []
    for df in measurements:
        X = df["X"].to_numpy()
        Y = qc.DPT(df["gThickness"].to_numpy(), N=dpt_N)
        s_b, s_e = qc.strip(X, Y)
        if mode == "min":
            peaks = qc.filt_peaks(-Y[s_b:s_e], filt=ampd_Z)
        elif mode == "max":
            peaks = qc.filt_peaks(Y[s_b:s_e], filt=ampd_Z)
        else:
            raise ValueError("invalid mode", mode)
        for p in (peaks + s_b):
            m_points.append((X[p], Y[p]))
    p_X = []
    p_Y = []
    for p_x, p_y in sorted(m_points, key=lambda x: x[0]):
        p_X.append(p_x)
        p_Y.append(p_y)
    return np.array(p_X), np.array(p_Y)

def thickness(measurements, dpt_N=3, ampd_Z=3.0):
    """Thickness model

    Input: list of dataframes of QC laser measurements

    Create a thickness curve along the length of the pipe
    Ignores structural pipe and socket/spigot details
    """
    return _thickness(measurements, dpt_N, ampd_Z,
                      mode="min")

def height(measurements, dpt_N=3, ampd_Z=3.0):
    """Height model

    Input: list of dataframes of QC laser measurements

    Create a height curve along the length of the pipe
    """
    return _thickness(measurements, dpt_N, ampd_Z,
                      mode="max")

def min_thickness(measurements):
    X, Y = thickness(measurements)
    Y = qc.operator_L(Y, 1)
    Y = qc.operator_L(Y, 2)
    return X, Y

def min_height(measurements):
    X, Y = height(measurements)
    Y = qc.operator_L(Y, 1)
    Y = qc.operator_L(Y, 2)
    return X, Y

def segmentX(measurements, mode="center"):
    """Segment boundaries on the X scale"""
    seg_bounds = []
    for df in measurements:
        X = df["X"].to_numpy()
        Y = qc.DPT(df["gThickness"].to_numpy(), N=3)
        s1, s2, s3 = qc.simple_segment(X, Y)
        seg_bounds.append([X[s1[0]], X[s1[1]], X[s3[0]], X[s3[1]]])
    seg_table = np.array(seg_bounds)
    offsets = seg_table[:,0].reshape((-1, 1))
    seg_table = seg_table - offsets
    if mode == "center":
        b = np.min(seg_table[:,0])
        e0 = np.min(seg_table[:,1])
        b2 = np.max(seg_table[:,2])
        e = np.max(seg_table[:,3])
        offset = np.min(offsets)
    else:
        raise ValueError("invalid mode", mode)
    return (b, e0, b2, e), offset

def profile(measurements, dpt_N=3, ampd_Z=3.0):
    """Profile model

    Input: list of dataframes of QC laser measurements

    Estimate the step between profile ribs
    """
    all_peaks = []
    for i, df in enumerate(measurements):
        X = df["X"].to_numpy()
        Y = qc.DPT(df["gThickness"].to_numpy(), N=dpt_N)
        s_b, s_e = qc.strip(X, Y)
        peaks = qc.filt_peaks(-Y[s_b:s_e], filt=ampd_Z)
        for p in (peaks + s_b):
            all_peaks.append((X[p], Y[p], i))

    rib_X = []
    curr_rib = []
    curr_measures = set()
    for p_x, p_y, i in sorted(all_peaks, key=lambda x: x[0]):
        if i in curr_measures:
            rib_X.append(np.mean(curr_rib))
            curr_rib = []
            curr_measures = set()
        curr_measures.add(i)
        curr_rib.append(p_x)
    if curr_rib:
        rib_X.append(np.mean(curr_rib))

    X = np.array(rib_X)
    Y = X[1:] - X[:-1]
    return X[1:], Y

def feature_select(X, y, k=10):
    """
    Select k best numerical features from X for binary prediction of y

    Inputs: X - numpy array
            y - sequence

    returns k f-value, column name pairs
    """
    f, _ = f_classif(X, y)
    ranking = np.argsort(-f)
    k = min(k, ranking.shape[0])
    return [(f[ranking[i]], ranking[i]) for
        i in range(k)]
