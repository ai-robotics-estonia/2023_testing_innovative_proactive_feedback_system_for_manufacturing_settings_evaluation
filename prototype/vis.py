#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import model

def plot_bar(var_data, title):
    vd = list(sorted(var_data, key=lambda x: x[2]))
    l = len(vd)
    x = np.arange(l)
    cmap = {0: "c", 1: "r", 2: "b"}
    color = [cmap[v[1]] for v in vd]
    plt.bar(x, [v[2] for v in vd], color=color)
    plt.xticks(x,
               [v[0] for v in vd],
               rotation=90)
    plt.suptitle(title)
    plt.show()

def plot_seg(X, y, seg_bounds):
    colors = ["r", "g", "b"]
    for (bi, ei), c in zip(seg_bounds, colors):
        plt.plot(X[bi:ei], y[bi:ei], c=c)
    plt.show()

class Plotter:
    def __init__(self, gen):
        self.gen = gen
        self.ymin = None
        self.ymax = None
        self.color_mode = "class"
        self.highlight = []

    def set_options(self, ymin=None, ymax=None, highlight=[]):
        self.ymin = ymin
        self.ymax = ymax
        self.highlight = highlight

    def plot_var(self, pipe_id, cl, series):
        X, Y, s_b, s_e = series
        c = "r" if cl else "c"
        if pipe_id in self.highlight:
            c = "b"
        plt.plot(X, Y[s_b:s_e], c=c)

    def plot_all(self):
        for pipe_id, cl, series in self.gen:
            self.plot_var(pipe_id, cl, series)
        if self.ymin is not None and self.ymax is not None:
            plt.ylim((self.ymin, self.ymax))
        plt.show()

class AggregatePlotter(Plotter):
    def plot_all(self):
        all_pipes = []
        for pipe_id, cl, (var, aggr, v) in self.gen:
            if pipe_id in self.highlight:
                cl = 2
            all_pipes.append((pipe_id, cl, v))
        plot_bar(all_pipes, var + " " + aggr)

class PeaksPlotter(Plotter):
    def plot_var(self, pipe_id, cl, series, peaks):
        X, Y, s_b, s_e = series
        plt.plot(X, Y[s_b:s_e], c="c")
        pX, pY = peaks
        plt.scatter(pX, pY, c="b")

    def plot_all(self):
        for pipe_id, cl, series, peaks in self.gen:
            self.plot_var(pipe_id, cl, series, peaks)
        if self.ymin is not None and self.ymax is not None:
            plt.ylim((self.ymin, self.ymax))
        plt.show()

class FeaturePlotter:
    def __init__(self, X, y, columns, aggr_mode=""):
        self.X = X
        self.y = y
        self.columns = columns
        self.aggr_mode = aggr_mode

    def plot_all(self):
        for fval, i in model.feature_select(self.X, self.y, k=15):
            plt.scatter(self.X[:, i], self.y)
            plt.suptitle("{} {} f={:.2f}".format(self.columns[i], self.aggr_mode, fval))
            plt.show()

    def pairs_plot(self, base_feature):
        col_idx = dict((c, i) for i, c in enumerate(self.columns))
        x = self.X[:, col_idx[base_feature]]
        colors = ["r" if y else "c" for y in self.y]
        for c in self.columns:
            if c in [base_feature, "QRSPipeId"]:
                continue
            plt.scatter(x, self.X[:, col_idx[c]], c=colors)
            plt.xlabel("{} {}".format(base_feature, self.aggr_mode))
            plt.ylabel("{} {}".format(c, self.aggr_mode))
            plt.show()
