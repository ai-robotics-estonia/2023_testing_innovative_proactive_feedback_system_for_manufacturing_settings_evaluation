#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def load_df(fn):
    return pd.read_csv(fn, header=0)

def normalize_df(df):
    #scaler = MinMaxScaler()
    scaler = StandardScaler()
    #scaler = QuantileTransformer()
    return pd.DataFrame(scaler.fit_transform(df),
                        columns=df.columns,
                        index=df.index)

def convert2dto3d(df, pipe_ids, natural=False):
    groupidx = df.groupby("QRSPipeId")
    df1 = df.loc[:, (df.columns != "TimeStep") & (df.columns != "QRSPipeId")]
    if not natural:
        df1 = normalize_df(df1)
    df2 = np.array([df1.loc[groupidx.indices[pipe_id], :] for pipe_id in pipe_ids])
    return df2

def load_classif(name="test.csv"):
    bla1 = load_df(name)
    bla2 = load_df(name + ".target")
    pipe_ids = bla2["QRSPipeId"].tolist()
    bla12 = convert2dto3d(bla1, pipe_ids)
    return bla12, bla2["y"].values

def load_regr(name="test4.csv"):
    bla1 = load_df(name)
    columns = [col for col in bla1.columns
               if col not in ["TimeStep", "QRSPipeId"]]
    bla2 = load_df(name + ".target")
    pipe_ids = [-1]
    for pipe_id in bla2["QRSPipeId"]:
        if pipe_id != pipe_ids[-1]:
            pipe_ids.append(pipe_id)
    pipe_ids = pipe_ids[1:]
    bla12 = convert2dto3d(bla1, pipe_ids)
    bla22 = convert2dto3d(bla2, pipe_ids, natural=True)
    print(bla12.shape, bla22.shape)
    return bla12, bla22, pipe_ids, columns

def load_livepreds(name):
    df = load_df(name)
    df.index = df["ID"]
    return df.to_dict("index")

SEED = 1001
def train_val(X, y):
    N = y.shape[0]
    n = int(N * 0.8)
    g = np.random.default_rng(seed=SEED)
    rndidx = g.permutation(range(N))
    y_t = y[rndidx[:n]].copy()
    y_v = y[rndidx[n:]].copy()
    X_t = X[rndidx[:n], :, :].copy()
    X_v = X[rndidx[n:], :, :].copy()
    return X_t, y_t, X_v, y_v
    
def change(X, win=15):
    """
    Compute change between time steps with a look-behind window
    """
    shifted = np.roll(X, win, axis=1)
    delta = X - shifted
    delta[:, :win, :] = 0
    return delta

def lookback(X, win=10):
    """
    return tensor A such that A[t,...] = X[t-win,...]
    representing "old" values of attributes/features
    
    for t<win, pad with dummy values
    """
    shifted = np.roll(X, win, axis=1)
    padding = X[:, 0, :].reshape((X.shape[0], -1, X.shape[2]))
    shifted[:, :win, :] = padding
    return shifted

def smooth(X, win=7):
    Xpad = np.pad(X, [(0, 0), (win - 1, 0), (0, 0)], mode="edge")
    return np.mean(np.lib.stride_tricks.sliding_window_view(Xpad, win, axis=1),
                  axis=3)

def dynamic_features(X, win1=3, win2=15, filt=7):
    """
    X is a tensor of (samples, timesteps, features)
    
    add 1st and 2nd order derivatives to time series
    
    1st order are not strictly derivatives but rather
    trend in a short window
    """
    d11 = change(X, win1)
    d12 = change(X, win2)
    if filt > 1:
        d11 = smooth(d11, filt)
        d12 = smooth(d12, filt)
    # d2 = change(d11, 1)
    # if filt > 1:
    #     d2 = smooth(d2, filt)
    # return np.concatenate([X, d11, d12, d2], axis=2)
    return np.concatenate([X, d11, d12], axis=2)

def temporal_features(X, steps=[5, 10, 25], filt=7):
    """
    X is a tensor of (samples, timesteps, features)
    
    adds features for attribute values in the past
    ("look-back")
    """
    hist = [X]
    for win in steps:
        feats = lookback(X, win)
        if filt > 1:
            feats = smooth(feats, filt)
        hist.append(X)
    return np.concatenate(hist, axis=2)

def flatten(X, y):
    l1 = X.shape[0] * X.shape[1]
    l2 = y.shape[0] * y.shape[1]
    if l1 != l2:
        raise ValueError
    l3 = X.shape[2]
    return X.reshape((l1, l3)), y.reshape((l2, 1))

def lasso():
    return Lasso(alpha=0.02)

def linreg():
    return LinearRegression()

def knn15():
    return KNeighborsRegressor(n_neighbors=15,
                               weights="uniform"
                               )

def knn7w():
    return KNeighborsRegressor(n_neighbors=7,
                               weights="distance"
                               )

def mlp():
    return MLPRegressor(hidden_layer_sizes=(32, 8),
                       activation="relu",
                       max_iter=2000)

def mlp3():
    return MLPRegressor(hidden_layer_sizes=(64, 16, 8),
                       activation="relu",
                       max_iter=2000)

def gbm():
    return HistGradientBoostingRegressor()

def median_baseline():
    return DummyRegressor(strategy="median")

def constant_baseline():
    return DummyRegressor(strategy="constant", constant=0.0)

def train_regr(model_func, X_t, y_t, kbest=None):
    X_t_flat, y_t_flat = flatten(X_t, y_t)
    if kbest is None:
        fsel = None
    else:
        fsel = SelectKBest(f_regression, k=kbest)
        fsel.fit(X_t_flat, y_t_flat.ravel())
        X_t_flat = fsel.transform(X_t_flat)

    model = model_func()
    model.fit(X_t_flat, y_t_flat.ravel())
    return model, fsel

def test_regr(model_conf, train_ds, test_ds, ds_name="default", detrend=True):
    X_t, y_t = train_ds
    X_v, y_v = test_ds
    if detrend:
        trend = np.mean(y_t, axis=0)
    else:
        trend = np.zeros(y_t.shape[1:])
    for label, model_func, kbest in model_conf:
        test_one(model_func, X_t, y_t, X_v, y_v, trend, kbest, label + " " + ds_name, detrend=detrend)

def test_one(model_func, X_t, y_t, X_v, y_v, trend, kbest, label, detrend=True):
    model, fsel = train_regr(model_func, X_t, y_t - trend, kbest)
    NN = X_v.shape[0]
    N = min(NN, 10)
    plot_h = N // 2
    plot_h += N % 2
    plot_w = 2
    t = np.arange(y_v.shape[1])
    fig, axs = plt.subplots(plot_h, plot_w)
    maes = []
    for i in range(NN):
        if fsel is None:
            X_test = X_v[i]
        else:
            X_test = fsel.transform(X_v[i])
        y_test = y_v[i]
        y_hat = model.predict(X_test).reshape((-1, 1))
        if detrend:
            y_hat = y_hat + trend
        mae = mean_absolute_error(y_test, y_hat)
        if i < N:
            u = i // 2
            v = i % 2
            axs[u, v].plot(t, y_test, "b")
            axs[u, v].plot(t, y_hat[:, 0], "r")
            if detrend:
                axs[u, v].plot(t, trend, "c--")
            axs[u, v].set_title("X[{}] mae {:.2f}".format(i, mae))
        maes.append(mae)

    plt.suptitle("{} mae {:.3f}".format(label, np.mean(maes)))
    plt.show()
    return maes

def iter_loocv(X, y, pipe_ids):
    N = X.shape[0]
    for i in range(N):
        X_t = np.vstack((X[:i, :, :], X[i + 1:, :, :]))
        y_t = np.vstack((y[:i, :, :], y[i + 1:, :, :]))
        #print(X_t.shape, y_t.shape)
        X_v = X[i:i + 1, :, :]
        y_v = y[i:i + 1, :, :]
        pipe_ids_v = pipe_ids[i:i + 1]
        yield X_t, y_t, X_v, y_v, pipe_ids_v

def iter_kfold(X, y, pipe_ids, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)
    for train, test in kf.split(X, y):
        pipe_ids_v = np.array(pipe_ids)[test]
        yield X[train], y[train], X[test], y[test], pipe_ids_v

def iter_once(train_ds, test_ds, pipe_ids_v):
    X_t, y_t = train_ds
    X_v, y_v = test_ds
    yield X_t, y_t, X_v, y_v, pipe_ids_v

def test_regr_shap(model_conf, test_gen, colnames, detrend=True):
    summary = {}
    for label, model_func, kbest in model_conf:

        model_stats = {}    # features can vary between CV slices,
                            # need dynamic structure to store
        for X_t, y_t, X_v, y_v, pipe_ids_v in test_gen:
            if detrend:
                trend = np.mean(y_t, axis=0)
            else:
                trend = np.zeros(y_t.shape[1:])
            model, fsel = train_regr(model_func, X_t, y_t - trend, kbest)

            print("SHAP batch", len(pipe_ids_v), "pipes")
            for j in range(X_v.shape[0]):
                if fsel is None:
                    X_test = X_v[j, :, :]
                    feature_names = colnames
                else:
                    X_test = fsel.transform(X_v[j, :, :])
                    supp = fsel.get_support()
                    feature_names = [col for i, col in enumerate(colnames)
                                     if supp[i]]

                if detrend:
                    # XXX: does not work currently, need to
                    # use masker somehow?
                    f = lambda x: model.predict(x) + trend[:, 0]
                else:
                    f = lambda x: model.predict(x)

                masker = shap.maskers.Independent(X_test)
                # explainer = shap.Explainer(f, masker,
                #                             feature_names=feature_names)
                explainer = shap.LinearExplainer(model, masker,
                                            feature_names=feature_names)
                shap_values = explainer(X_test)
                # Fast explainer for for GBM, detrend=False
                # explainer = shap.TreeExplainer(model, masker,
                #                             feature_names=feature_names)
                # shap_values = explainer(X_test,
                #                         check_additivity=False)
                # shap.plots.beeswarm(shap_values)
                # shap.plots.bar(shap_values)
                vals = shap_values.abs.mean(0).values

                pipe_stats = {}
                for i, fn in enumerate(feature_names):
                    if fn in pipe_stats:
                        pipe_stats[fn] += vals[i]
                    else:
                        pipe_stats[fn] = vals[i]
                model_stats[pipe_ids_v[j]] = pipe_stats
                # print(vals[np.argsort(-vals)])
                # print(int(np.max(vals)), pipe_ids_v[j])
                # print(np.array(feature_names)[np.argsort(-vals)])

        summary[label] = model_stats

    return summary

def test_regr_cv(model_conf, test_gen, ds_name="default", detrend=True):
    maes = {}
    for k, _, _ in model_conf:
        maes[k] = []

    for X_t, y_t, X_v, y_v, _ in test_gen:
        if detrend:
            trend = np.mean(y_t, axis=0)
        else:
            trend = np.zeros(y_t.shape[1:])
        for label, model_func, kbest in model_conf:
            model, fsel = train_regr(model_func, X_t, y_t - trend, kbest)
            X_test, y_test = flatten(X_v, y_v)
            if fsel is not None:
                X_test = fsel.transform(X_test)
            y_hat = model.predict(X_test).reshape((-1, 1))
            if detrend:
                y_hat = y_hat + np.tile(trend, (X_v.shape[0], 1))
            mae = mean_absolute_error(y_test, y_hat)
            maes[label].append(mae)
    
    df = pd.DataFrame()
    for label, m in maes.items():
        print(label, np.mean(m))
        df[label] = np.array(m)

    sns.kdeplot(df).set(title=ds_name)

def test_livepred(model_conf, test_gen, lp_target, detrend=True):
    summary = {}
    for label, model_func, kbest in model_conf:
        p_times = []
        p_class = []
        p_target = []
        plot_data = []
        pipe_ids = []
        for X_t, y_t, X_v, y_v, pipe_ids_v in test_gen:
            if detrend:
                trend = np.mean(y_t, axis=0)
            else:
                trend = np.zeros(y_t.shape[1:])

            model, fsel = train_regr(model_func, X_t, y_t - trend, kbest)
            for j in range(X_v.shape[0]):
                if fsel is None:
                    X_test =X_v[j, :, :]
                else:
                    X_test = fsel.transform(X_v[j, :, :])
                y_hat = model.predict(X_test).reshape((-1, 1))
                if detrend:
                    y_hat = y_hat + trend
                # print(np.mean(y_v[j]))
                p_time, pred, truth, plot_stats = test_live_one(y_hat, pipe_ids_v[j], lp_target)
                p_times.append(p_time)
                p_class.append(pred)
                p_target.append(truth)
                plot_data.append(plot_stats)
            pipe_ids += pipe_ids_v.tolist()

        # plot of FP, TP
        df = pd.DataFrame()
        df["p_time"] = np.array(p_times)
        df["pred"] = np.array(p_class)
        df["correct"] = (np.array(p_class) == np.array(p_target))
        sns.kdeplot(data=df[df["pred"]], x="p_time", hue="correct", cut=0).set(title=label)

        df["pipe_id"] = np.array(pipe_ids)
        df["target"] = np.array(p_target)
        summary[label] = df

        # confusion matrix
        dp = ConfusionMatrixDisplay.from_predictions(p_target, p_class, cmap="Blues")
        dp.ax_.set_title(label)

        # predictions sample
        plot_live_sample(plot_data, label)

    return summary

LIVEPRED_SMOOTH = 20
LIVEPRED_BURNIN = 100
def test_live_one(y_hat, pipe_id, lp_target):
    min_v = lp_target[pipe_id]["min_allowed"]
    qc_v = lp_target[pipe_id]["mean_height"]
    is_fault = qc_v < (min_v - 0.0001)

    # median smoothing of model noise
    # warn of any part of pipe dropping below tolerance
    # y_pad = np.pad(y_hat.ravel(), (LIVEPRED_SMOOTH - 1, 0), mode="constant",
    #                constant_values=(min_v + 2.0, 0))
    # y_running = np.median(np.lib.stride_tricks.sliding_window_view(y_pad, LIVEPRED_SMOOTH),
    #                axis=1)

    # cumulative mean
    # warn of mean height of pipe dropping below tolerance
    y_running = np.cumsum(y_hat) / (np.arange(y_hat.shape[0]) + 1)

    # burn in
    y_running[:LIVEPRED_BURNIN] = min_v + 0.0001

    bad_ts = np.argwhere(y_running < min_v)
    #print(np.mean(y_hat), np.min(y_running), qc_v, min_v)
    if bad_ts.shape[0] > 1:
        pred = True
        p_ts = np.min(bad_ts)
    else:
        pred = False
        p_ts = y_hat.shape[0]
    p_time = p_ts / y_hat.shape[0]
    return p_time, pred, is_fault, (y_hat, p_ts, min_v)

def plot_live_sample(plot_data, label):
    N = min(len(plot_data), 10)
    plot_h = N // 2
    plot_h += N % 2
    plot_w = 2
    T = np.arange(plot_data[0][0].shape[0])
    fig, axs = plt.subplots(plot_h, plot_w)
    for i in range(N):
        y_hat, bad_ts, min_v = plot_data[i]
        u = i // 2
        v = i % 2
        t = np.arange(bad_ts)
        axs[u, v].plot(T, np.full(T.shape, min_v), "b")
        axs[u, v].plot(t, y_hat[:bad_ts, 0], "r")
        if bad_ts < T.shape[0]:
            axs[u, v].plot([bad_ts], [min_v], "ro", markersize=8)
        axs[u, v].set_title("X[{}]".format(i))

    plt.suptitle("{}".format(label))
    plt.show()

def prep_ds(X, y, win1=3, win2=15, dyn=False, temporal=True, split=True):
    # deprecated code to shift the entire feature set in time
    # replaced by temporal feature generation
    # if timeshift > 0:
    #     # delay effect of features in X
    #     X = X[:, :-timeshift, :]
    #     y = y[:, timeshift:, :]
    # elif timeshift < 0:
    #     # effect of features in X appears sooner
    #     X = X[:, -timeshift:, :]
    #     y = y[:, :timeshift, :]

    if dyn:
        if temporal:
            raise ValueError("temporal and dynamic not supported together (would cause cartesian product of features)")
        X_f = dynamic_features(X, win1, win2)
    elif temporal:
        X_f = temporal_features(X)
    else:
        X_f = X
    if split:
        X_t, y_t, X_v, y_v = train_val(X_f, y)
    else:
        X_t, y_t = X_f, y
        X_v, y_v = None, None
    return (X_t, y_t), (X_v, y_v)

def feat_vs_target(train_ds, test_ds, trange=None):
    X_t, y_t = train_ds
    X_v, y_v = test_ds
    N = X_t.shape[2]
    if trange is None:
        X_t_flat, y_t_flat = flatten(X_t, y_t)
        X_v_flat, y_v_flat = flatten(X_v, y_v)
    else:
        X_t_flat, y_t_flat = flatten(X_t[:, trange[0]:trange[1], :],
                                     y_t[:, trange[0]:trange[1], :])
        X_v_flat, y_v_flat = flatten(X_v[:, trange[0]:trange[1], :],
                                     y_v[:, trange[0]:trange[1], :])
    M1 = X_t_flat.shape[0]
    M2 = X_v_flat.shape[0]
    hue = np.zeros(M1 + M2, dtype="int")
    hue[M1:] = 1
    target = np.concatenate((y_t_flat[:, 0], y_v_flat[:, 0]))
    df = pd.DataFrame()
    df["y"] = target
    df["set"] = hue
    for i in range(N):
        var = np.concatenate((X_t_flat[:, i], X_v_flat[:, i]))
        label = "feature {}".format(i)
        df[label] = var
        sns.jointplot(df, x=label, y="y", hue="set")

def feat_plot(ds, feat, pipe):
    X, y = ds
    t = np.arange(X.shape[1])
    plt.plot(t, X[pipe, :, feat])
    plt.show()

TOPN = 3
PLOT_CLASSES = [("True positive", (True, True)),
                ("False negative", (False, True)),
                ("False positive", (True, False))]

def shap_summary(shap_vals, pred_stats):
    def add(d, feature, val):
        if feature in d:
            d[feature] += val
        else:
            d[feature] = val

    def report(d):
        for cl, totals in d.items():
            print("==", cl)
            for i, (k, v) in enumerate(sorted(
                                totals.items(), key=lambda x: -x[1])):
                if i < TOPN:
                    print("  ", k, v)

    def plot(d, counts={}, nfeat=5):
        overall = {}
        for _, totals in d.items():
            for k, v in totals.items():
                add(overall, k, v)
        selected = [feature for feature, _ in
                    sorted(overall.items(), key=lambda x: -x[1])][:nfeat]
        df = pd.DataFrame.from_records(
            [{"feature": feature,
              "class": cl,
              "val": d[clkey].get(feature, 0) / counts.get(clkey, 1)}
                 for feature in selected
                     for cl, clkey in PLOT_CLASSES
             ])
        plot = sns.barplot(df, x="feature", y="val", hue="class")
        plot.set_xticklabels(plot.get_xticklabels(),
                             rotation=45, horizontalalignment="right")

    for label in shap_vals:
        cumulative = {(False, False) : {}, (False, True) : {},
                      (True, False) : {},(True, True) : {},}
        count = {(False, False) : {}, (False, True) : {},
                      (True, False) : {},(True, True) : {},}
        # XXX: map from column names properly
        classes = dict((r[3], (r[1], r[4]))
                       for r in pred_stats[label].itertuples(index=False))
        class_counts = {(False, False) : 0, (False, True) : 0,
                      (True, False) : 0, (True, True) : 0}
        for pipe_id in shap_vals[label]:
            pipe_shap = [(k, v)
                         for k, v in shap_vals[label][pipe_id].items()]
            for i, (k, v) in enumerate(sorted(
                                pipe_shap, key=lambda x: -x[1])):
                add(cumulative[classes[pipe_id]], k, v)
                if i < TOPN:
                    add(count[classes[pipe_id]], k, 1)
            class_counts[classes[pipe_id]] += 1

        # print(label)
        # report(cumulative)
        # report(count)
        plot(count, class_counts, nfeat=10)
        # plot(cumulative, nfeat=10)
    
TEST_MODELS = [
    # ("Lasso", lasso, None),
    # ("Baseline med", median_baseline, None),
    # ("Baseline const", constant_baseline, None),
    ("GBM", gbm, None),
    # ("GBM 15 features", gbm, 15),
    # ("GBM 30 features", gbm, 30),
    #("MLP 32+8", mlp, None),
    # ("MLP 64+16+8", mlp3, None),
    # ("Linear 15 features", linreg, 15),
    # ("Linear 8 features", linreg, 8),
    # ("k-NN k=15 15 features", knn15, 15),
    # ("k-NN k=15", knn15, None),
    ]

if __name__ == "__main__":
    # X, y, pipe_ids, columns = load_regr("test8.csv")
    # X, y, pipe_ids, _ = load_regr("test8_med.csv")
    # X, y, pipe_ids, _ = load_regr("thickness_6k.csv")
    # X, y, pipe_ids, columns = load_regr("height_6k.csv")
    # live_target = load_livepreds("height_6k_livepred.csv")
    # X, y, pipe_ids, columns = load_regr("test11.csv")
    # live_target = load_livepreds("test11_livepred.csv")
    # X, y, _, _ = load_regr("height_delta_vanad.csv")
    # X2, y2, pipe_ids, _ = load_regr("height_delta_uued.csv")
    # live_target = load_livepreds("height_delta_uued_livepred.csv")
    X, y, pipe_ids, columns = load_regr("test14.csv")
    live_target = load_livepreds("test14_livepred.csv")

    # train_ds, test_ds = prep_ds(X, y, dyn=False, temporal=False)
    # train_ds, test_ds = prep_ds(X, y)
    train_ds, _ = prep_ds(X, y, split=False)

    # test_ds, _ = prep_ds(X2, y2, split=False)

    # feat_vs_target(train_ds, test_ds)
    # feat_vs_target(train_ds, test_ds, (250, 975))
    # test_regr(TEST_MODELS, train_ds, test_ds, detrend=True)
    # test_regr_cv(TEST_MODELS,
    #               iter_kfold(train_ds[0], train_ds[1], pipe_ids),
    #               # iter_once(train_ds, test_ds, pipe_ids),
    #               detrend=True)
    pred_stats = test_livepred(TEST_MODELS,
                    iter_kfold(train_ds[0], train_ds[1], pipe_ids),
                    # iter_once(train_ds, test_ds, pipe_ids),
                  live_target, detrend=False)
    # shap_vals = test_regr_shap(TEST_MODELS,
    #                 iter_kfold(train_ds[0], train_ds[1], pipe_ids),
    #               columns * 4, detrend=False)
    # shap_summary(shap_vals, pred_stats)
