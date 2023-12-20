#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict
import csv
import os.path
import re
import qc
import log
import model

BASE_DIR = "AIRE & TalTech andmed"
QC_DIR = "Andmed kvaliteedipingilt"
LOG_DIR = "Tootmise parameetriline info"
EXP_DIR = "Tootmise materjalide info"

class Loader:
    QC_META = [
        "lg:data.bogie2Cal",
        "lg:data.bogie3Cal",
        "lg:data.bogie1EndIndex",
        "lg:data.bogie2EndIndex",
        "lg:data.bogie3EndIndex",
        "lg:data.weight"
    ]

    def __init__(self, fn, base_dir=BASE_DIR):
        self.fn = fn
        self.base_dir = base_dir

    def make_indexed_df(self, vals, idx, keys, Xname=None):
        if Xname is None:
            Xname = keys[0]
        cols = []
        for k in keys:
            cols.append(pd.Series(vals[k], index=idx[k], name=k))
        df = pd.concat(cols, axis=1)
        df["X"] = df[Xname]
        return df

    def load_triples(self, infile):
        vals = defaultdict(list)
        idx = defaultdict(list)
        meta = {}
        with open(infile) as f:
            rd = csv.reader(f)
            for r in rd:
                if r[1] in self.QC_META:
                    if r[2] == "REAL":
                        v = float(r[3])
                    elif r[2] == "REAL":
                        v = int(r[3])
                    else:
                        v = r[3]
                    meta[r[1]] = v
                elif r[2] == "REAL":
                    m = re.match("([^\[]+)\[([0-9]+)\]", r[1])
                    if m is not None:
                        vals[m.group(1)].append(float(r[3]))
                        idx[m.group(1)].append(int(m.group(2)))

        out = []
        out.append(self.make_indexed_df(vals, idx,
                                   ["lg:data.actPosiB1_2",
                                    "lg:data.bogie1DSE",
                                    "lg:data.bogie2Dimetix",
                                    "lg:data.bogie2DSE",
                                    "gThickness"],
                                    "lg:data.actPosiB1_2"
                                   ))
        out.append(self.make_indexed_df(vals, idx,
                                   ["lg:data.actPosiB3",
                                    "lg:data.bogie3Dimetix",
                                    "lg:data.bogie3DSE"],
                                    "lg:data.actPosiB3"))
        out.append(meta)
        return out

    def pipe_meta(self, infile):
        return pd.read_csv(infile, header=0)

    def pipe_log(self, infile):
        return pd.read_csv(infile, header=0, delimiter=";")

    def one_pipe(self, exp_df, pipe_id):
        df = exp_df[exp_df["QRS Pipe ID"] == pipe_id]
        out = {}
        if df.shape[0] == 1:
            out["exp"] = df
            for scan in [0, 1, 2]:
                scanfile = os.path.join(self.base_dir, QC_DIR,
                                        "id{}_{}.csv".format(pipe_id, scan))
                out["s{}".format(scan)] = self.load_triples(scanfile)
            guid = df["PipeData GUID"].tolist()[0]
            logfile = os.path.join(self.base_dir, LOG_DIR,
                                   "{}.csv".format(guid))
            out["log"] = self.pipe_log(logfile)

        return out

    def get_meta(self):
        infile = os.path.join(self.base_dir, EXP_DIR, self.fn)
        return self.pipe_meta(infile)

    def pipe_iter(self, filter_id=[], pipe_ids=[]):
        exp_df = self.get_meta()
        for _, row in exp_df.iterrows():
            pipe_id = row["QRS Pipe ID"]
            if pipe_id in filter_id:
                continue
            if pipe_ids and pipe_id not in pipe_ids:
                continue
            cl = row["Is Scrap"]
            pipe_data = self.one_pipe(exp_df, pipe_id)
            yield pipe_id, cl, pipe_data

    def load_pipe(self, pipe_id):
        exp_df = self.get_meta()
        return self.one_pipe(exp_df, pipe_id)

class Extractor:
    def __init__(self, ld):
        self.ld = ld
        self.var = None
        self.var_type = None
        self.model_func = None
        self.xaxis = None

    def set_options(self, segment=-1, filter_id=[],
                  end_detect="extruder"):
        self.filter_id = filter_id
        self.segment = segment
        self.end_detect = end_detect

    def do_gen(self, pipe_ids=[]):
        self.pipe_ids = pipe_ids
        return self.gen_all()

    def iter_attr(self, attr, pipe_ids=[]):
        self.var_type = "attr"
        self.var = attr
        return self.do_gen(pipe_ids)

    def iter_feature(self, feature, pipe_ids=[]):
        self.var_type = "feature"
        self.var = feature
        return self.do_gen(pipe_ids)

    def iter_model(self, model_func, pipe_ids=[]):
        self.var_type = "model"
        self.model_func = model_func
        self.var = str(model_func.__name__)
        return self.do_gen(pipe_ids)

    def seg_bounds(self, pipe_data, y_size, T):
        if self.segment in [0, 1, 2]:
            b, e = log.detect_ends(pipe_data["log"])
            segX, _ = model.segmentX([
                pipe_data["s{}".format(i)][0] for i in range(2)
                ])
            segT = log.segments(log.attr_interp(pipe_data["log"], "CarriagePosition", T),
                                T[b], T[e], segX)
            s_b, s_e = segT[self.segment]
        elif self.end_detect in ["extruder", "coretube", "carriage"]:
            b, e = log.detect_ends(pipe_data["log"],
                                   method=self.end_detect)
            s_b = T[b]
            s_e = T[e]
        else:
            s_b = 0
            s_e = y_size
        return s_b, s_e

    def prep_XY(self, pipe_id, cl, pipe_data):
        T, _ = log.ref_axes(pipe_data["log"])
        if self.var_type == "attr":
            Y = log.attr_interp(pipe_data["log"],
                                self.var, T)
        elif self.var_type == "feature":
            feat_df = log.make_features(pipe_data["log"], pipe_data["exp"])
            Y = feat_df[self.var]
        elif self.var_type == "model":
            Xm, Y = self.model_func(pipe_id, cl, pipe_data, T=T)
        else:
            raise NotImplementedError("unknown variable type")

        s_b, s_e = self.seg_bounds(pipe_data, Y.shape[0], T)

        if self.var_type == "model":
            X = Xm[s_b:s_e]
        elif self.xaxis is None:
            X = np.arange(s_e - s_b)
        else:
            X = log.attr_interp(pipe_data["log"],
                                self.xaxis, T)[s_b:s_e]
        return X, Y, s_b, s_e

    def gen_all(self):
        for pipe_id, cl, pipe_data in self.ld.pipe_iter(filter_id=self.filter_id,
                                                        pipe_ids=self.pipe_ids):
            try:
                X, Y, s_b, s_e = self.prep_XY(pipe_id, cl, pipe_data)
            except:
                print(pipe_id, "failed to extract data series")
                continue
            yield pipe_id, cl, (X, Y, s_b, s_e)

class AggregateExtractor(Extractor):
    def __init__(self, ld):
        super().__init__(ld)
        self.aggr_mode = "median"

    def gen_all(self):
        for pipe_id, cl, pipe_data in self.ld.pipe_iter(filter_id=self.filter_id,
                                                        pipe_ids=self.pipe_ids):
            X, Y, s_b, s_e = self.prep_XY(pipe_id, cl, pipe_data)
            if self.aggr_mode == "mean":
                v = np.mean(Y[s_b:s_e])
            elif self.aggr_mode == "sum":
                v = np.sum(Y[s_b:s_e])
            elif self.aggr_mode == "median":
                v = np.median(Y[s_b:s_e])
            else:
                raise NotImplementedError("unknown mode")
            yield pipe_id, cl, (self.var, self.aggr_mode, v)

class MeasurementExtractor(Extractor):
    def __init__(self, ld):
        super().__init__(ld)
        self.measurement = 0
        self.filter_width = 4

    def prep_XY(self, pipe_id, cl, pipe_data):
        s = pipe_data["s{}".format(self.measurement)]
        X = None
        Y = None
        if self.var_type == "attr":
            for df in s[:1]:
                if self.var in df.columns:
                    X = df["X"]
                    Y = qc.DPT(df[self.var].to_numpy(),
                               N=self.filter_width)
                    break
        elif self.var_type == "model":
            X, Y = self.model_func(pipe_id, cl, pipe_data)
        else:
            raise NotImplementedError("unsupported variable type")
        if X is None:
            raise ValueError("unknown variable")

        if self.var_type == "model":
            b, e = 0, X.shape[0]
        else:
            b, e = qc.strip(X, Y)
        # print("measurement", pipe_id, X.shape, Y.shape, X[0], Y[0], X[-1], Y[-1], b, e)
        if self.segment in [0, 1, 2]:
            segX, segX_offs = model.segmentX([
                pipe_data["s{}".format(i)][0] for i in range(2)
                ])
            X_b = segX[self.segment] + segX_offs
            X_e = segX[self.segment + 1] + segX_offs
            s_b = max(b, b + np.min(np.argwhere(np.array(X[b:e]) > X_b)))
            s_e = min(e, b + np.max(np.argwhere(np.array(X[b:e]) < X_e)))
            offs = X_b
        else:
            s_b, s_e = b, e
            offs = X[s_b]
        X = X - offs

        # print("post segment", offs, X[s_b], X[s_e], s_b, s_e)
        return X[s_b:s_e], Y, s_b, s_e

class PeaksExtractor(MeasurementExtractor):
    def gen_all(self):
        for pipe_id, cl, pipe_data in self.ld.pipe_iter(filter_id=self.filter_id,
                                                        pipe_ids=self.pipe_ids):
            X, Y, s_b, s_e = self.prep_XY(pipe_id, cl, pipe_data)

            if self.peaks == "upper":
                peaks = qc.filt_peaks(Y[s_b:s_e],
                                      filt=self.peak_filt)
            elif self.peaks == "lower":
                peaks = qc.filt_peaks(-Y[s_b:s_e],
                                      filt=self.peak_filt)
            else:
                raise NotImplementedError("invalid peak type", self.peaks)
    
            pX = X[(peaks + s_b)]
            pY = Y[peaks]
            yield pipe_id, cl, (X, Y, s_b, s_e), (pX, pY)

SKIP_ATTRIBUTES = ["CarriagePosition"]
SKIP_PATT = ["Temperature"]
class FeatureExtractor(Extractor):
    def all_features(self, pipe_ids=[]):
        self.pipe_ids = pipe_ids
        return self._all_features()

    def skip(self, feature):
        if feature in SKIP_ATTRIBUTES:
            return True
        for p in SKIP_PATT:
            if re.search(p, feature):
                return True
        return False

    def setup_target(self):
        self.y = []

    def update_target(self, pipe_id, cl, pipe_data, s_b, s_e):
        self.y.append(cl)

    def get_target(self):
        return self.y

    def _all_features(self):
        """
        Tabular features from pipe data

        Returns X - table of features
                y - class (binary)
                columns - column headers for X
        """
        self.setup_target()
        xseries = defaultdict(list)
        pipe_ids = []
        for pipe_id, cl, pipe_data in self.ld.pipe_iter(filter_id=self.filter_id,
                                                        pipe_ids=self.pipe_ids):
            T, _ = log.ref_axes(pipe_data["log"])
            s_b, s_e = self.seg_bounds(pipe_data, T[-1], T)

            for i in range(1, pipe_data["log"].shape[1]):
                col_name = pipe_data["log"].columns[i]
                if self.skip(col_name):
                    continue
                v = log.attr_seq_feature(pipe_data["log"], col_name,
                                    s_b, s_e, self.aggr_mode, T)
                xseries[col_name].append(v)

            feat_df = log.make_features(pipe_data["log"], pipe_data["exp"])
            for col_name in feat_df.columns:
                v = log.seq_feature(feat_df[col_name],
                                    s_b, s_e, self.aggr_mode)
                xseries[col_name].append(v)

            self.update_target(pipe_id, cl, pipe_data, s_b, s_e)
            pipe_ids.append(pipe_id)

        y = self.get_target()
        X = [pipe_ids]
        columns = ["QRSPipeId"]
        for k, v in xseries.items():
            if len(v) == len(y):
                X.append(v)
                columns.append(k)
        X = np.array(X).T
        return X, y, columns

    def save_features(self, pipe_ids, filename):
        X, y, columns = self.all_features(pipe_ids)
        df = pd.DataFrame(data=X, columns=columns)
        df["y"] = pd.Series(y)
        df.to_csv(filename, encoding='utf-8', index=False)

class RegressionExtractor(FeatureExtractor):
    def __init__(self, ld, target_gen):
        super().__init__(ld)
        self.aggr_mode = "median"
        self.target_gen = target_gen

    def setup_target(self):
        self.precalc_y = {}
        for pipe_id, cl, series in self.target_gen:
            _, Y, s_b, s_e = series
            if self.aggr_mode == "median":
                v = np.median(Y[s_b:s_e])
            else:
                raise NotImplementedError("unknown mode")
            self.precalc_y[pipe_id] = v
        self.y = []

    def update_target(self, pipe_id, cl, pipe_data, s_b, s_e):
        self.y.append(self.precalc_y[pipe_id])

class SeriesClassifExtractor(FeatureExtractor):
    def update_target(self, pipe_id, cl, pipe_data, s_b, s_e, Xref):
        self.y.append((pipe_id, cl))

    def get_target(self, series_length):
        return pd.DataFrame(self.y, columns=["QRSPipeId", "y"])

    def target_exists(self, pipe_id):
        return True

    def _all_features(self):
        """
        Time series features from pipe data

        Returns X - dataframe of timeseries, each row has pipe_id, timestep
                y - dataframe of class (binary), each row has pipe_id
        """
        # collect series
        self.setup_target()
        xseries = defaultdict(dict)
        pipe_ids = []
        min_series_length = -1
        for pipe_id, cl, pipe_data in self.ld.pipe_iter(filter_id=self.filter_id,
                                                        pipe_ids=self.pipe_ids):
            if not self.target_exists(pipe_id):
                continue
            T, Xsparse = log.ref_axes(pipe_data["log"])
            Xref = log.time_interp(T, Xsparse)
            try:
                s_b, s_e = self.seg_bounds(pipe_data, T[-1], T)
            except:
                print(pipe_id, "pipe segmentation failed")
                continue
            if self.truncate_min > 0:
                series_length = s_e - s_b
                if series_length < self.truncate_min:
                    continue
                min_series_length = max(min_series_length, series_length)

            for i in range(1, pipe_data["log"].shape[1]):
                col_name = pipe_data["log"].columns[i]
                if self.skip(col_name):
                    continue
                v = log.attr_interp(pipe_data["log"], col_name, T)
                xseries[col_name][pipe_id] = v[s_b:s_e].tolist()

            feat_df = log.make_features(pipe_data["log"], pipe_data["exp"])
            for col_name in feat_df.columns:
                v = feat_df[col_name]
                xseries[col_name][pipe_id] = v[s_b:s_e].tolist()

            self.update_target(pipe_id, cl, pipe_data, s_b, s_e, Xref)
            pipe_ids.append(pipe_id)

        # valid series and length
        #columns = ["QRSPipeId", "TimeStep"]
        columns = []
        for k, v in xseries.items():
            if len(v) == len(pipe_ids):
                columns.append(k)
                if self.truncate_min > 0:
                    for vv in v.values():
                        min_series_length = min(len(vv), min_series_length)
            else:
                print(k, "missing from some pipes")
        print("Data shape", len(columns), "x", min_series_length)

        # construct the X dataframe
        cols = []
        pipe_id_series = []
        ts_series = []
        for pipe_id in pipe_ids:
            if self.truncate_min > 0:
                series_length = min_series_length
            else:
                series_length = len(xseries[columns[0]][pipe_id])
            pipe_id_series += [pipe_id] * series_length
            ts_series += list(range(series_length))
        cols.append(pd.Series(pipe_id_series, name="QRSPipeId"))
        cols.append(pd.Series(ts_series, name="TimeStep"))
        for col_name in columns:
            v = []
            for pipe_id in pipe_ids:
                if self.truncate_min > 0:
                    v += xseries[col_name][pipe_id][:min_series_length]
                else:
                    v += xseries[col_name][pipe_id]
            cols.append(pd.Series(v, name=col_name))

        X = pd.concat(cols, axis=1)
        y = self.get_target(min_series_length)
        return X, y

    def save_features(self, pipe_ids, filename):
        X, y = self.all_features(pipe_ids)
        X.to_csv(filename, encoding="utf-8", index=False)
        y.to_csv(filename + ".target", encoding="utf-8", index=False)

class SeriesRegressionExtractor(SeriesClassifExtractor):
    def __init__(self, ld, target_gen):
        super().__init__(ld)
        self.target_gen = target_gen

    def setup_target(self):
        self.precalc_y = {}
        for pipe_id, cl, series in self.target_gen:
            X, Y, s_b, s_e = series
            self.precalc_y[pipe_id] = (np.array(X),
                                       np.array(Y[s_b:s_e]))
        self.y = []

    def target_exists(self, pipe_id):
        return pipe_id in self.precalc_y

    def update_target(self, pipe_id, cl, pipe_data, s_b, s_e, Xref):
        tX, tY = self.precalc_y[pipe_id]
        # T = np.arange(s_e - s_b)
        # print(tX.shape, tY.shape, T.shape)
        # ratio = (T[-1] / tX[-1])
        # self.y.append((pipe_id, np.interp(T, tX * ratio, tY)))
        y = np.interp(Xref[s_b:s_e] - Xref[s_b], tX, tY)
        print(y.shape, Xref[s_e] - Xref[s_b], tX[0], tX[-1])
        self.y.append((pipe_id, y))

    def get_target(self, min_series_length):
        y = []
        pipe_id_series = []
        ts_series = []
        for pipe_id, v in self.y:
            if self.truncate_min > 0:
                series_length = min_series_length
            else:
                series_length = len(v)
            pipe_id_series += [pipe_id] * series_length
            ts_series += list(range(series_length))
            y += v[:series_length].tolist()
        Y = pd.concat([pd.Series(pipe_id_series, name="QRSPipeId"),
                       pd.Series(ts_series, name="TimeStep"),
                       pd.Series(y, name="y")], axis=1)
        return Y

class SegmentExtractor(Extractor):
    def gen_all(self):
        for pipe_id, cl, pipe_data in self.ld.pipe_iter(filter_id=self.filter_id,
                                                        pipe_ids=self.pipe_ids):
            T, _ = log.ref_axes(pipe_data["log"])
            # bounds from segment detection
            b, e = log.detect_ends(pipe_data["log"], method="extruder")
            segX, _ = model.segmentX([
                pipe_data["s{}".format(i)][0] for i in range(2)
                ])
            segT = log.segments(log.attr_interp(pipe_data["log"], "CarriagePosition", T),
                                T[b], T[e], segX)
            s_b1, s_e1 = segT[1]

            # bounds from production log
            # b2, e2 = log.detect_ends(pipe_data["log"], method="coretube")
            # s_b2 = T[b2]
            # s_e2 = T[e2]

            guid = pipe_data["exp"]["PipeData GUID"].to_list()[0]
            base_ts = log.start_time(pipe_data["log"])
            yield (pipe_id, guid, base_ts, T[b], s_b1, s_e1, T[e])

    def save_segments(self, pipe_ids, filename):
        self.pipe_ids = pipe_ids
        rows = []
        for pipe_id, guid, base_ts, pipe_b, s_b1, s_e1, pipe_e in self.gen_all():
            rows.append([pipe_id,
                   guid,
                   base_ts + pd.Timedelta(seconds=pipe_b),
                   base_ts + pd.Timedelta(seconds=s_b1),
                   base_ts + pd.Timedelta(seconds=s_e1),
                   base_ts + pd.Timedelta(seconds=pipe_e)])
        df = pd.DataFrame(data=rows,
            columns=["QRS Pipe ID", "GUID", "Pipe start", "Profile start (algo1)", "Profile end (algo1)", "Pipe end"])
        df.to_csv(filename, encoding='utf-8', index=False)
