#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loading and visualization of Krah pipe data

Pipe data directory structure:

Tootmise materjalide info/export.csv
   \
    - QRS Pipe ID - pipe id 31160
   \
    - PipeData GUID - production log id a749a172-7e0c-4a7d-8173-890da6455164

Tootmise parameetriline info/a749a172-7e0c-4a7d-8173-890da6455164.csv

Andmed kvaliteedipingilt/id31160_0.csv
Andmed kvaliteedipingilt/id31160_1.csv
Andmed kvaliteedipingilt/id31160_2.csv

"""

import numpy as np
import qc
import log
import model
import vis
import data

def segment_iter(pipe_data):
    for measurement in [0, 1, 2]:
        s = pipe_data["s{}".format(measurement)][0]
        pts = s["lg:data.actPosiB1_2"]
        y = s["gThickness"]
        yield pts, y, qc.simple_segment(pts, y)

def show_segments(ld, pipe_ids):
    for pipe_id in pipe_ids:
        pipe_data = ld.load_pipe(pipe_id)
        for pts, y, seg_bounds in segment_iter(pipe_data):
            vis.plot_seg(pts, y, seg_bounds)

def one_pipe_size(pipe_id, pipe_data):
    T, X = log.ref_axes(pipe_data["log"])
    b, e = log.detect_ends(pipe_data["log"])
    qrs_l = []
    for _, _, seg_bounds in segment_iter(pipe_data):
        qrs_l.append(seg_bounds[2][1] - seg_bounds[0][0])
    print(pipe_id, "Length (production)", X[e] - X[b],
          "Length (QRS) {:.2f}".format(np.mean(qrs_l)))

def show_size(ld, filter_id, pipe_ids=[]):
    for pipe_id, _, pipe_data in ld.pipe_iter(filter_id=filter_id,
                                           pipe_ids=pipe_ids):
        one_pipe_size(pipe_id, pipe_data)

def deposit_model(pipe_id, cl, pipe_data, T=None):
    if T is None:
        T, _ = log.ref_axes(pipe_data["log"])
    X = log.attr_interp(pipe_data["log"],
                        "CarriagePosition", T)
    feat_df = log.make_features(pipe_data["log"], T=T)
    Y = model.interp_deposit(X,
                      feat_df["ExtruderTotal"],
                      feat_df["RotationFactor"],
                      kernel_w=100)
    return X, Y

def deposit_simple(pipe_id, cl, pipe_data, T=None):
    if T is None:
        T, _ = log.ref_axes(pipe_data["log"])
    X = log.attr_interp(pipe_data["log"],
                        "CarriagePosition", T)
    feat_df = log.make_features(pipe_data["log"], T=T)
    Y1 = feat_df["ExtruderTotal"].to_numpy()
    Y2 = feat_df["RotationFactor"].to_numpy()
    Y = Y1 / (Y2 + 0.0001)
    return X, Y

def thickness_model(pipe_id, cl, pipe_data, T=None):
    X, Y = model.thickness([
        pipe_data["s{}".format(i)][0] for i in range(2)
        ])
    return X, Y

def min_thickness_model(pipe_id, cl, pipe_data, T=None):
    X, Y = model.min_thickness([
        pipe_data["s{}".format(i)][0] for i in range(2)
        ])
    return X, Y

def min_height_model(pipe_id, cl, pipe_data, T=None):
    X, Y = model.min_height([
        pipe_data["s{}".format(i)][0] for i in range(2)
        ])
    return X, Y

def height_delta_model(pipe_id, cl, pipe_data, T=None):
    X, Y = model.min_height([
        pipe_data["s{}".format(i)][0] for i in range(2)
        ])
    target = pipe_data["exp"]["Nõutav välisseina paksus"].tolist()[0]
    return X, Y - target

def ribs_model(pipe_id, cl, pipe_data, T=None):
    X, Y = model.profile([
        pipe_data["s{}".format(i)][0] for i in range(2)
        ])
    return X, Y

MODELS = {"deposit" : deposit_model,
          "deposit_simple" : deposit_simple,
          "thickness" : thickness_model,
          "min_thickness" : min_thickness_model,
          "ribs" : ribs_model,
          "min_height" : min_height_model,
          "height_delta" : height_delta_model
    }
def get_generator(ld, args):
    if args.m == "plot":
        xtr = data.Extractor(ld)
        if args.X:
            xtr.xaxis = args.X
    elif args.m == "aggr":
        xtr = data.AggregateExtractor(ld)
        xtr.aggr_mode = args.s
    elif args.m in ["shape", "ts_regr"]:
        if args.P in ["upper", "lower"]:
            xtr = data.PeaksExtractor(ld)
            xtr.peaks = args.P
            xtr.peak_filt = 3.0
        else:
            xtr = data.MeasurementExtractor(ld)
        xtr.measurement = args.e
        xtr.filter_width = args.w
    elif args.m == "shape_regr":
        xtr = data.MeasurementExtractor(ld)
        xtr.measurement = args.e
        xtr.filter_width = args.w

    xtr.set_options(segment=args.S,
                   filter_id=args.f,
                   end_detect=args.d)
    if args.F:
        g = xtr.iter_feature(args.F, pipe_ids=args.p)
    elif args.A:
        g = xtr.iter_attr(args.A, pipe_ids=args.p)
    elif args.M:
        m_func = MODELS.get(args.M)
        if m_func is not None:
            g = xtr.iter_model(m_func, pipe_ids=args.p)
        else:
            raise ValueError("unknown model", args.M)
    else:
        print("Nothing to show, use -A or -F or -M")
    return g

def run_plot(ld, args):
    g = get_generator(ld, args)
    if args.m == "plot":
        p = vis.Plotter(g)
    elif args.m == "aggr":
        p = vis.AggregatePlotter(g)
        p.aggr_mode = args.s
    elif args.m == "shape":
        if args.P in ["upper", "lower"]:
            p = vis.PeaksPlotter(g)
        else:
            p = vis.Plotter(g)

    p.set_options(highlight=args.l)
    if args.y and len(args.y) == 2:
        p.ymin = args.y[0]
        p.ymax = args.y[1]
    p.plot_all()

def extractor_args(xtr, args):
    xtr.set_options(segment=args.S,
                   filter_id=args.f,
                   end_detect=args.d)
    xtr.aggr_mode = args.s
    xtr.truncate_min = args.t
    return xtr

def feature_detect(ld, args):
    xtr = extractor_args(data.FeatureExtractor(ld), args)
    X, y, columns = xtr.all_features(args.p)
    p = vis.FeaturePlotter(X, y, columns, aggr_mode=args.s)
    p.plot_all()

def feature_scatter(ld, args, base_feature):
    xtr = extractor_args(data.FeatureExtractor(ld), args)
    X, y, columns = xtr.all_features(args.p)
    p = vis.FeaturePlotter(X, y, columns, aggr_mode=args.s)
    p.pairs_plot(base_feature)

def feature_dump(ld, args):
    xtr = extractor_args(data.FeatureExtractor(ld), args)
    xtr.save_features(args.p, args.o)

def regr_feature_dump(ld, args):
    target_gen = get_generator(ld, args)
    xtr = extractor_args(data.RegressionExtractor(ld, target_gen),
                         args)
    xtr.save_features(args.p, args.o)

def ts_feature_dump(ld, args):
    xtr = extractor_args(data.SeriesClassifExtractor(ld), args)
    xtr.save_features(args.p, args.o)

def ts_regr_feature_dump(ld, args):
    target_gen = get_generator(ld, args)
    xtr = extractor_args(data.SeriesRegressionExtractor(ld, target_gen),
                         args)
    xtr.save_features(args.p, args.o)

def segment_dump(ld, args):
    xtr = extractor_args(data.SegmentExtractor(ld), args)
    xtr.save_segments(args.p, args.o)


# -b '../AIRE & TalTech andmed' -m plot -A CarriageSpeed -i export_merged2.csv
# -b '../PR90-011.57 RG' -m plot -A CarriageSpeed -i export_merged.csv
# -b '../PR90-011.57 RG' -m aggr -s median -F CoretubeStep -d coretube -i export_merged_ribivahe.csv
# -b '../PR90-011.57 RG' -m shape -M ribs -i export_merged_ribivahe.csv -p 31127 31130 31131 31090 31089 30671 30661 31053 31040
# -b '../PR90-011.57 RG' -m shape -M min_height -i export_merged_h_madal.csv -p 31528 31204 31126 31127 30871 -l 30871 -y 88 97
# -b '../PR90-011.57 RG' -m feature -s median -i export_merged_h_madal.csv -d coretube -o '../test.csv'

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    # p.add_argument('-t', action="store_true",
    #     help="truncate timeseries to same length (ts_classif, ts_regr modes)")
    p.add_argument('-t', type=int, default=-1,
        help="truncate timeseries to same length (ts_classif, ts_regr modes). Argument is min length")
    p.add_argument('-p', type=int, nargs="*", default=[],
        help="show only pipe id")
    p.add_argument('-f', type=int, nargs="*", default=[],
        help="filter pipe id")
    p.add_argument('-l', type=int, nargs="*", default=[],
        help="highlight pipe id")
    p.add_argument('-m', type=str, default="plot",
        help="run mode (plot,segment,size,aggr,shape,feature,shape_regr,scatter,ts_classif,ts_regr)")
    p.add_argument('-F', type=str, default="",
        help="feature name")
    p.add_argument('-A', type=str, default="",
        help="attribute name")
    p.add_argument('-M', type=str, default="",
        help="model name")
    p.add_argument('-X', type=str, default="",
        help="X-axis attribute name")
    p.add_argument('-S', type=int, default=-1,
        help="select segment (0, 1, 2, default: none)")
    p.add_argument('-P', type=str, default="",
        help="show peaks (upper/lower/none), default none")
    p.add_argument('-s', type=str, default="sum",
        help="aggregate function (sum/mean/median)")
    p.add_argument('-e', type=int, default=0,
        help="measurement number")
    p.add_argument('-d', type=str, default="",
        help="end detection")
    p.add_argument('-b', type=str, default=data.BASE_DIR,
        help="base directory")
    p.add_argument('-i', type=str, default="",
        help="input file (pipe metadata)")
    p.add_argument('-o', type=str, default="",
        help="output file")
    p.add_argument('-w', type=int, default=4,
        help="pulse filter width")
    p.add_argument('-y', type=float, nargs=2, default=[],
        help="y range (-y min max)")
    args = p.parse_args()
    if not args.i:
        print("Use -i <pipe metadata>")
        import sys
        sys.exit(1)

    ld = data.Loader(args.i, args.b)

    if args.m in ["plot", "shape", "aggr"]:
        run_plot(ld, args)
    elif args.m == "segment":
        if args.o:
            segment_dump(ld, args)
        elif args.p:
            show_segments(ld, args.p)
        else:
            print("-m segment requires -p <pipe_id>")
    elif args.m == "feature":
        if args.o:
            feature_dump(ld, args)
        else:
            feature_detect(ld, args)
    elif args.m == "scatter":
        if args.F:
            base_feature = args.F
        else:
            base_feature = args.A
        if not base_feature:
            raise ValueError("need -A or -F with scatter")
        feature_scatter(ld, args, base_feature)
    elif args.m == "shape_regr":
        if args.o:
            regr_feature_dump(ld, args)
        else:
            raise NotImplementedError
    elif args.m == "ts_classif":
        if args.o:
            ts_feature_dump(ld, args)
        else:
            raise NotImplementedError
    elif args.m == "ts_regr":
        if args.d:
            print("-d option not supported with regression target")
        elif args.o:
            ts_regr_feature_dump(ld, args)
        else:
            raise NotImplementedError
    elif args.m == "size":
        show_size(ld, filter_id=args.f,
                  pipe_ids=args.p)
