#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def ref_axes(log_df):
    """Create reference axes for the production log
    
    returns T, X
    T - time index: int array (seconds)
    X - carriage position: float array (mm)
    """
    dt = pd.to_datetime(log_df["Timestamp"],
                format="%m/%d/%Y %H:%M:%S", errors="coerce")
    T = np.array(dt.view(dtype = "int64")) // 1000000000
    shift = T.min()
    T = T - shift
    X = log_df["CarriagePosition"].to_numpy()
    return (T, X)

def start_time(log_df):
    """Time of start of the log

    returns pandas timestamp for second 0 of log timeseries
    """
    dt = pd.to_datetime(log_df["Timestamp"],
                format="%m/%d/%Y %H:%M:%S", errors="coerce")
    return dt[0]

def time_interp(T, Y):
    """Interpolate using time axis
    T - array of int (with gaps)
    Y - measured value
    
    returns Y interpolated at points (0, 1, 2, ..., max(T))
    """
    T_new = np.arange(T[-1] + 1)
    return np.interp(T_new, T, Y)

def attr_interp(log_df, attr_name, T=None):
    """Interpolate an attribute using time axis
    
    returns attribute value at every second
    """
    if T is None:
        T, _ = ref_axes(log_df)
    Y = log_df[attr_name].to_numpy()
    return time_interp(T, Y)

def attr_seq_feature(log_df, attr_name, s_b, s_e, mode, T=None):
    """
    Calculate a feature from a log data timeseries

    mode - median/sum
    s_b, s_e - section of the log (time indices)
    """
    y = attr_interp(log_df, attr_name, T)
    return seq_feature(y, s_b, s_e, mode)

def seq_feature(y, s_b, s_e, mode):
    """
    Calculate a feature from a timeseries

    mode - median/sum
    s_b, s_e - section of the series (time indices)
    """
    if mode == "median":
        v = np.median(y[s_b:s_e])
    elif mode == "sum":
        v = np.sum(y[s_b:s_e])
    else:
        raise NotImplementedError("unknown mode")
    return v

def detect_ends(log_df, method="extruder"):
    """Estimate the part of production log
    where material is deposited on the pipe
    
    returns b, e
    indices are for the non-interpolated array
    use T[b], T[e] to get indices for time-interpolated arrays
    """
    if method == "extruder":
        e0 = log_df["ExtruderDetails[0].IsOn"].to_numpy()
        e1 = log_df["ExtruderDetails[1].IsOn"].to_numpy()
        e2 = log_df["ExtruderDetails[2].IsOn"].to_numpy()
        e_ind = np.argwhere((e0 == 1) | (e1 == 1) | (e2 == 1))
        b = np.min(e_ind)
        e = np.max(e_ind)
    elif method == "coretube":
        ct = log_df["CoretubeSpeed"].to_numpy()
        e_ind = np.argwhere(ct > 1)
        b = np.min(e_ind)
        e = np.max(e_ind)
    elif method == "carriage":
        cp = log_df["CarriagePosition"].to_numpy()
        d1 = cp[1:] - cp[:-1]
        change = np.pad(np.sum(np.lib.stride_tricks.sliding_window_view(d1, 3),
                                axis=1),
                        (3, 0),
                        mode="edge")
        fwd_ind = np.argwhere((change > 10) & (cp < 100))
        rev_ind = np.argwhere(change < -10)
        # forward movement over multiple measurements near pos 0
        b = np.min(fwd_ind)
        # first reverse after continuous forward movement
        e = np.min(rev_ind[rev_ind > (b + 100)])
    else:
        raise NotImplementedError("method", method, "not implemented")
    return b, e

# Deprecation candidates:
#
# def spigot_timewindow(X, b, e, w=300):
#     """Estimate the time window for making
#     last w mm of the pipe
    
#     X - interpolated carriage position
#     b, e - detected interpolated end positions (use detect_ends())
    
#     returns b, e
#     indices are for the interpolated array
#     """
#     maxi = np.argmax(X[b:e]) + b
#     maxX = X[maxi]
#     minX = maxX - w
#     s_ind = np.argwhere(X < minX)
#     filt_s_ind = s_ind[s_ind < maxi]
#     s_b = np.max(filt_s_ind)
#     return s_b, e

# def socket_timewindow(X, b, e, w=300):
#     """Estimate the time window for making
#     the first w mm of the pipe

#     X - interpolated carriage position
#     b, e - detected interpolated end positions (use detect_ends())

#     returns b, e
#     indices are for the interpolated array
#     """
#     mini = np.argmin(X[b:e]) + b
#     minX = X[mini]
#     maxX = minX + w
#     s_ind = np.argwhere(X > maxX)
#     filt_s_ind = s_ind[s_ind > mini]
#     s_e = np.min(filt_s_ind)
#     return b, s_e

def segments(X, b, e, segX):
    """Map segment time windows to production log

    X - interpolated carriage position
    b, e - detected interpolated end positions

    segX - segment bound X coordinates from QC measurements

    returns (b0, e0), (b1, e1), (b2, e2)
    indices are for the interpolated array,
    equivalent to full seconds
    """
    x12 = segX[3]
    x22 = np.max(X[b:e])
    offs = x22 - x12
    tf_segX = np.array(segX) + offs

    b0 = b + np.min(np.argwhere(X[b:e] > tf_segX[0]))
    b1 = b + np.min(np.argwhere(X[b:e] > tf_segX[1]))
    b2 = b + np.min(np.argwhere(X[b:e] > tf_segX[2]))
    e2 = b + np.max(np.argwhere(X[b:e] < tf_segX[3]))

    b1 = max(b0 + 2, b1)
    b2 = max(b1 + 2, b2)
    return (b0, b1-1), (b1, b2-1), (b2, e2)

FEATURES = ["ExtruderTotal",
            "SurfaceCover",
            "SurfaceStep",
            "CoretubeStep",
            "MaterialStretch",
            "CoretubeStretch",
            "MaterialInverse",
            # "Test000",
            # "Test001",
            # "Test010",
            # "Test011",
            # "Test100",
            # "Test101",
            # "Test110",
            # "Test111",
            "NormCarriageSpeed",
            "NormRotationSpeed",
            "NormExtruderPR0",
            "NormExtruderPR1",
            "ToolTempRange1",
            "ToolTempRange2",
            "ExtruderZones0",
            "ExtruderZones2"
            ]
EPSILON = 0.0001
def make_features(log_df, exp_df, features=FEATURES, T=None):
    """Feature engineering from production attributes
    
    returns a dataframe
    by default, features are time-interpolated"""
    if T is None:
        T, _ = ref_axes(log_df)
    out = []

    series_cache = {}
    def get_series(s_name):
        s = series_cache.get(s_name)
        if s is None:
            s = attr_interp(log_df, s_name, T)
            series_cache[s_name] = s
        return s

    for feat in features:
        if feat == "ExtruderTotal":
            tot = None
            for attr in ["ExtruderDetails[0].Throughput",
                         "ExtruderDetails[1].Throughput",
                         "ExtruderDetails[2].Throughput"]:
                s = get_series(attr)
                if tot is None:
                    tot = s
                else:
                    tot = tot + s
            out.append(pd.Series(tot, name=feat))
            series_cache[feat] = tot
        elif feat == "SurfaceCover":
            s1 = get_series("RotationSpeed")
            s2 = get_series("CarriageSpeed")
            s = 1 / ((s1 * s2) + EPSILON)
            out.append(pd.Series(s, name=feat))
        elif feat == "SurfaceStep":
            s1 = get_series("CarriageSpeed")
            s2 = get_series("RotationSpeed")
            s = s1 / (s2 + EPSILON)
            out.append(pd.Series(s, name=feat))
        elif feat == "CoretubeStep":
            s1 = get_series("CarriageSpeed")
            s2 = get_series("CoretubeSpeed")
            s = s1 / (s2 + EPSILON)
            out.append(pd.Series(s, name=feat))
        elif feat == "MaterialStretch":
            s1 = get_series("ExtruderDetails[1].Throughput")
            s2 = get_series("CoretubeSpeed")
            s = s2 / (s1 + EPSILON)
            out.append(pd.Series(s, name=feat))
        elif feat == "CoretubeStretch":
            s1 = get_series("RotationSpeed")
            s2 = get_series("CoretubeSpeed")
            s = s1 / (s2 + EPSILON)
            out.append(pd.Series(s, name=feat))
        elif feat == "NormCarriageSpeed":
            s1 = get_series("CarriageSpeed")
            diam = exp_df["Diameeter"].tolist()[0]
            s = s1 * diam
            out.append(pd.Series(s, name=feat))
        elif feat == "NormRotationSpeed":
            s1 = get_series("RotationSpeed")
            diam = exp_df["Diameeter"].tolist()[0]
            s = s1 * diam
            out.append(pd.Series(s, name=feat))
        elif feat == "NormExtruderPR0":
            s1 = get_series("ExtruderDetails[0].Throughput")
            prof = exp_df["Nõutav välisseina paksus"].tolist()[0]
            s = s1 / prof
            out.append(pd.Series(s, name=feat))
        elif feat == "NormExtruderPR1":
            s1 = get_series("ExtruderDetails[1].Throughput")
            prof = exp_df["Nõutav välisseina paksus"].tolist()[0]
            s = s1 / prof
            out.append(pd.Series(s, name=feat))
        elif feat == "MaterialInverse":
            s1 = get_series("RotationSpeed")
            s2 = get_series("CarriageSpeed")
            s3 = get_series("ExtruderTotal")
            s = (s1 * s2) / (s3 + EPSILON)
            out.append(pd.Series(s, name=feat))
        elif feat.startswith("ToolTempRange"):
            attrs = []
            if feat[-1] == "1":
                temps = ["0", "1", "2", "3", "4", "5",
                         "6", "7", "8", "9", "10", "11"]
            elif feat[-1] == "2":
                temps = ["12", "13", "14", "15", "16"]
            else:
                raise ValueError
            for t in temps:
                attrs.append(get_series("ToolTemperatures[{}]".format(t)))
            s = attrs[0]
            for attr in attrs[1:]:
                s = s + attr
            s = s / len(attrs)
            out.append(pd.Series(s, name=feat))
        elif feat.startswith("ExtruderZones"):
            attrs = []
            temps = ["0", "1", "2", "3", "4", "5"]
            xtr = feat[-1]
            for t in temps:
                attrs.append(get_series(
                    "ExtruderDetails[{}].TemperatureZones[{}]".format(
                        xtr, t)))
            s = attrs[0]
            for attr in attrs[1:]:
                s = s + attr
            s = s / len(attrs)
            out.append(pd.Series(s, name=feat))
        elif feat.startswith("Test"):
            attrs = [get_series("RotationSpeed"),
                     get_series("CarriageSpeed"),
                     get_series("ExtruderTotal")]
            a = np.ones(attrs[0].shape[0])
            b = np.ones(attrs[0].shape[0])
            for i in range(len(attrs)):
                c = feat[4 + i]
                if c == "0":
                    a = a * attrs[i]
                else:
                    b = b * attrs[i]
            s = a / (b + EPSILON)
            out.append(pd.Series(s, name=feat))
        else:
            raise NotImplementedError("feature", feat, "not implemented")
    return pd.concat(out, axis=1)
