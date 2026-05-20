# -*- coding: utf-8 -*-
"""
NIR classification - LITE v2

(A) FULL SPECTRUM
(B) FULL-SPECTRUM WINDOW SEARCH
(C) miniNIR-like WINDOW SEARCH

For Doenca and Estadio:
- TRAIN + TEST metrics for ALL models
- overtones removed
- full spectrum comparison added
- best-model plots for each block
- joblib saved for blocks B and C
"""

import os
import re
import shutil
import unicodedata
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

warnings.filterwarnings("ignore")

CSV_PATH = "Compostas (Novas) - Espetros-Doenca e estadios.csv"
OUT_DIR = "NIR_classification_outputs_lite_v2"

OUTER_SPLITS = 3
INNER_SPLITS = 2

TEST_SIZE = 0.20
RANDOM_STATE = 42
VERBOSE = True

WIN_WIDTH_FULL = 500
STEP_FULL = 300
TOPK_WINDOWS_FULL = 3
MIN_FEATURES_FULL = 50

MININIR_HALF_WIDTH = 150
TOPK_WINDOWS_MININIR = 3
MIN_FEATURES_MININIR = 20

SAVGOL_WINDOW = 21
SAVGOL_POLY = 2
SAVGOL_DERIV = 1


def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)

def _clean(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s

def read_csv_robust(path):
    p = Path(path)
    if p.is_dir():
        csvs = list(p.glob("*.csv")) or list(p.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in directory: {path}")
        p = csvs[0]
        log(f"Using CSV file: {p}")
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(p, sep=";", decimal=",", low_memory=False, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(p, sep=";", decimal=",", low_memory=False)

def find_label_col(df, candidates):
    cleaned_map = {_clean(c): c for c in df.columns}
    for ck, orig in cleaned_map.items():
        for cand in candidates:
            if cand in ck:
                return orig
    return None

def parse_spectral_columns(df, exclude_keywords=("doenca","doença","estadio","estádio","conc","class","label","type","unnamed")):
    cols = []
    axis = []
    for c in df.columns:
        ck = _clean(c)
        if any(k in ck for k in exclude_keywords):
            continue
        s = str(c).strip().replace(",", ".")
        try:
            v = float(s)
            cols.append(c); axis.append(v)
        except Exception:
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", str(c))
            if m:
                v = float(m.group(1).replace(",", "."))
                cols.append(c); axis.append(v)
    if not cols:
        raise ValueError("Could not detect spectral columns.")
    idx = np.argsort(axis)
    cols_sorted = [cols[i] for i in idx]
    axis_sorted = np.array([axis[i] for i in idx], dtype=float)
    return cols_sorted, axis_sorted

def coerce_numeric(df, cols):
    out = df[cols].copy()
    for c in cols:
        if out[c].dtype.kind in "biufc":
            continue
        s = out[c].astype(str).str.strip()
        s = s.str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^0-9\.\-eE+]", "", regex=True)
        out[c] = pd.to_numeric(s, errors="coerce")
    return out

def slice_window_by_axis(X, axis_vals, start, end):
    lo, hi = (start, end) if start <= end else (end, start)
    mask = (axis_vals >= lo) & (axis_vals <= hi)
    idx = np.where(mask)[0]
    return X[:, idx], idx

def nm_to_cm1(nm):
    return 1e7 / float(nm)

def make_full_windows(axis_vals):
    axis_min, axis_max = float(axis_vals.min()), float(axis_vals.max())
    windows = []
    start = axis_min
    while start + WIN_WIDTH_FULL <= axis_max:
        end = start + WIN_WIDTH_FULL
        windows.append((float(start), float(end)))
        start += STEP_FULL
    return windows

def make_mininir_windows():
    led_nm = [1300, 1400, 1600, 1700, 1900, 2100]
    centers = [nm_to_cm1(nm) for nm in led_nm]
    windows = []
    for c in centers:
        windows.append((float(c - MININIR_HALF_WIDTH), float(c + MININIR_HALF_WIDTH)))
    cmin, cmax = min(centers), max(centers)
    windows.append((float(cmin - MININIR_HALF_WIDTH), float(cmax + MININIR_HALF_WIDTH)))
    return windows


class SavGolSNV(BaseEstimator, TransformerMixin):
    def __init__(self, window=SAVGOL_WINDOW, poly=SAVGOL_POLY, deriv=SAVGOL_DERIV):
        self.window = int(window)
        self.poly = int(poly)
        self.deriv = int(deriv)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]
        w = int(self.window)
        if w % 2 == 0:
            w += 1
        if w > n_feat:
            w = n_feat if n_feat % 2 == 1 else max(3, n_feat - 1)
        if w < 3:
            Xf = X.copy()
        else:
            Xf = savgol_filter(X, window_length=w, polyorder=min(self.poly, w - 1), deriv=self.deriv, axis=1)
        mu = Xf.mean(axis=1, keepdims=True)
        sd = Xf.std(axis=1, keepdims=True) + 1e-12
        return (Xf - mu) / sd


def metrics_binary(y_true, y_pred, y_proba=None):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_proba)) if y_proba is not None else np.nan
    }

def metrics_multiclass(y_true, y_pred, y_proba=None):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["auc_ovr_weighted"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            out["auc_ovr_weighted"] = np.nan
    else:
        out["auc_ovr_weighted"] = np.nan
    return out

def tune_threshold_for_recall(y_true, proba_pos):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_rec = 0.5, -np.inf
    for t in thresholds:
        yhat = (proba_pos >= t).astype(int)
        rec = recall_score(y_true, yhat, pos_label=1, zero_division=0)
        if rec > best_rec:
            best_rec = rec
            best_t = float(t)
    return best_t, best_rec

def plot_roc_binary(y_true, proba_pos, out_png, title):
    fpr, tpr, _ = roc_curve(y_true, proba_pos)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_roc_multiclass_ovr_macro(y_true, proba, n_classes, out_png, title):
    Y = label_binarize(y_true, classes=np.arange(n_classes))
    fpr, tpr = {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], proba[:, i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(all_fpr, mean_tpr, label=f"Macro AUC={macro_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_pca_test_only(X_train_proc, X_test_proc, y_test, out_png_test, title_prefix):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(X_train_proc)
    Zte = pca.transform(X_test_proc)
    pc1 = float(pca.explained_variance_ratio_[0] * 100)
    pc2 = float(pca.explained_variance_ratio_[1] * 100)

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in np.unique(y_test):
        idx = (y_test == c)
        ax.scatter(Zte[idx, 0], Zte[idx, 1], label=str(c), alpha=0.8)
    ax.set_title(f"{title_prefix} | Test")
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.legend(title="Classe", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png_test, dpi=200)
    plt.close(fig)
    return pc1, pc2

def save_confusion_matrix(y_true, y_pred, labels, tick_labels, out_png, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def plot_score_vs_windows(df_rank, task_label, tag, out_dir):
    labels = df_rank["window_label"].tolist()
    x = np.arange(len(labels))
    y = df_rank["nested_cv_mean"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o")
    N = max(1, len(labels) // 12)
    ax.set_xticks(x[::N])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), N)], rotation=45, ha="right")
    ax.set_xlabel("Spectral window (cm$^{-1}$)")
    ax.set_ylabel("Nested-CV score")
    ax.set_title(f"{task_label} | window ranking | {tag}")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"WinRankPlot_{task_label}_{tag}.png"), dpi=200)
    plt.close(fig)

def grids_lite():
    g = {
        "SVM_RBF": (
            SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}
        ),
        "RF": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "clf__n_estimators": [300],
                "clf__max_depth": [None, 10],
                "clf__min_samples_leaf": [1, 3],
                "clf__max_features": ["sqrt"],
            }
        ),
    }
    if HAS_XGB:
        g["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                n_estimators=300,
                learning_rate=0.05,
            ),
            {
                "clf__max_depth": [3, 4],
                "clf__min_child_weight": [1, 5],
                "clf__subsample": [0.7],
                "clf__colsample_bytree": [0.7],
                "clf__reg_lambda": [1.0, 2.0],
            }
        )
    return g

def nested_cv_score_for_window(Xw, y, task):
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = "recall" if task == "doenca" else "recall_macro"
    pipe = Pipeline([
        ("pre", SavGolSNV()),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE))
    ])
    grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}
    fold_scores = []
    for tr_idx, te_idx in outer.split(Xw, y):
        X_tr, X_te = Xw[tr_idx], Xw[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1)
        gs.fit(X_tr, y_tr)
        yhat = gs.best_estimator_.predict(X_te)
        if task == "doenca":
            s = recall_score(y_te, yhat, pos_label=1, zero_division=0)
        else:
            s = recall_score(y_te, yhat, average="macro", zero_division=0)
        fold_scores.append(s)
    return float(np.mean(fold_scores)), float(np.std(fold_scores))

def rank_windows(X, axis_vals, y, task, mode):
    if mode == "FULL_SEARCH":
        windows = make_full_windows(axis_vals)
        min_feats = MIN_FEATURES_FULL
    elif mode == "MININIR":
        windows = make_mininir_windows()
        min_feats = MIN_FEATURES_MININIR
    else:
        raise ValueError(f"Unknown mode: {mode}")

    rows = []
    total = len(windows)
    for i, (start, end) in enumerate(windows, start=1):
        Xw, idx = slice_window_by_axis(X, axis_vals, start, end)
        if Xw.shape[1] < min_feats:
            continue
        log(f"[RANK {task} | {mode}] Window {i}/{total}: {start:.0f}-{end:.0f} cm-1 | n_features={Xw.shape[1]}")
        mean_s, std_s = nested_cv_score_for_window(Xw, y, task)
        rows.append({
            "mode": mode,
            "task": task,
            "start_cm-1": float(start),
            "end_cm-1": float(end),
            "window_label": f"{float(start):.0f}-{float(end):.0f}",
            "n_features": int(Xw.shape[1]),
            "nested_cv_mean": mean_s,
            "nested_cv_std": std_s
        })
    if not rows:
        raise ValueError(f"No valid windows for mode={mode}")
    return pd.DataFrame(rows).sort_values("nested_cv_mean", ascending=False).reset_index(drop=True)

def evaluate_all_models_nested(X, y, task):
    model_grids = grids_lite()
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring_inner = "recall" if task == "doenca" else "recall_macro"

    rows = []
    best_name, best_primary = None, -np.inf

    for name, (clf, grid) in model_grids.items():
        log(f"    [MODEL {task}] {name}")
        y_true_all, y_pred_all, proba_all = [], [], []
        fold_scores = []

        for tr_idx, te_idx in outer.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            pipe = Pipeline([
                ("pre", SavGolSNV()),
                ("scaler", StandardScaler()),
                ("clf", clf)
            ])
            gs = GridSearchCV(pipe, grid, scoring=scoring_inner, cv=inner, n_jobs=-1)
            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_

            yhat = best.predict(X_te)
            y_true_all.append(y_te)
            y_pred_all.append(yhat)

            proba = None
            try:
                proba = best.predict_proba(X_te)
            except Exception:
                proba = None

            if task == "doenca":
                s = recall_score(y_te, yhat, pos_label=1, zero_division=0)
                if proba is not None:
                    proba_all.append(proba[:, 1])
            else:
                s = recall_score(y_te, yhat, average="macro", zero_division=0)
                if proba is not None:
                    proba_all.append(proba)
            fold_scores.append(s)

        mean_s = float(np.mean(fold_scores))
        std_s = float(np.std(fold_scores))
        y_true_cat = np.concatenate(y_true_all)
        y_pred_cat = np.concatenate(y_pred_all)

        if task == "doenca":
            proba_cat = np.concatenate(proba_all) if len(proba_all) else None
            m = metrics_binary(y_true_cat, y_pred_cat, proba_cat)
            primary = m["recall_pos"]
        else:
            proba_cat = np.concatenate(proba_all, axis=0) if len(proba_all) else None
            m = metrics_multiclass(y_true_cat, y_pred_cat, proba_cat)
            primary = m["recall_macro"]

        row = {"model": name, "Train_nestedCV_mean_score": mean_s, "Train_nestedCV_std_score": std_s}
        row.update(m)
        rows.append(row)

        if primary > best_primary:
            best_primary = primary
            best_name = name

    df_models = pd.DataFrame(rows).sort_values("Train_nestedCV_mean_score", ascending=False).reset_index(drop=True)
    return df_models, best_name

def fit_and_test_all_models(Xtr, ytr, Xte, yte, task):
    df_models, best_name = evaluate_all_models_nested(Xtr, ytr, task)
    rows = []

    for _, r in df_models.iterrows():
        name = str(r["model"])
        clf, grid = grids_lite()[name]
        pipe = Pipeline([
            ("pre", SavGolSNV()),
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])
        inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scoring = "recall" if task == "doenca" else "recall_macro"
        gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1)
        gs.fit(Xtr, ytr)

        best_pipe = gs.best_estimator_
        thr = 0.5

        if task == "doenca":
            proba_oof = cross_val_predict(best_pipe, Xtr, ytr, cv=inner, method="predict_proba", n_jobs=-1)[:, 1]
            thr, _ = tune_threshold_for_recall(ytr, proba_oof)

        best_pipe.fit(Xtr, ytr)
        yhat = best_pipe.predict(Xte)

        proba = None
        try:
            proba = best_pipe.predict_proba(Xte)
        except Exception:
            proba = None

        if task == "doenca":
            proba_pos = proba[:, 1] if proba is not None else None
            if proba_pos is not None:
                yhat = (proba_pos >= thr).astype(int)
            test_m = metrics_binary(yte, yhat, proba_pos)
            rows.append({
                "model": name,
                "Train_nestedCV_mean_score": float(r["Train_nestedCV_mean_score"]),
                "Train_nestedCV_std_score": float(r["Train_nestedCV_std_score"]),
                "Train_accuracy": float(r["accuracy"]),
                "Train_precision": float(r["precision_pos"]),
                "Train_recall": float(r["recall_pos"]),
                "Train_f1": float(r["f1_pos"]),
                "Train_specificity": float(r["specificity"]),
                "Train_auc": float(r["auc"]) if pd.notna(r["auc"]) else np.nan,
                "Test_accuracy": float(test_m["accuracy"]),
                "Test_precision": float(test_m["precision_pos"]),
                "Test_recall": float(test_m["recall_pos"]),
                "Test_f1": float(test_m["f1_pos"]),
                "Test_specificity": float(test_m["specificity"]),
                "Test_auc": float(test_m["auc"]) if pd.notna(test_m["auc"]) else np.nan,
                "best_estimator": best_pipe,
                "test_pred": yhat,
                "test_proba": proba_pos,
                "threshold": thr,
                "is_best_train": int(name == best_name),
            })
        else:
            test_m = metrics_multiclass(yte, yhat, proba)
            rows.append({
                "model": name,
                "Train_nestedCV_mean_score": float(r["Train_nestedCV_mean_score"]),
                "Train_nestedCV_std_score": float(r["Train_nestedCV_std_score"]),
                "Train_accuracy": float(r["accuracy"]),
                "Train_precision": float(r["precision_macro"]),
                "Train_recall": float(r["recall_macro"]),
                "Train_f1": float(r["f1_macro"]),
                "Train_specificity": np.nan,
                "Train_auc": float(r["auc_ovr_weighted"]) if pd.notna(r["auc_ovr_weighted"]) else np.nan,
                "Test_accuracy": float(test_m["accuracy"]),
                "Test_precision": float(test_m["precision_macro"]),
                "Test_recall": float(test_m["recall_macro"]),
                "Test_f1": float(test_m["f1_macro"]),
                "Test_specificity": np.nan,
                "Test_auc": float(test_m["auc_ovr_weighted"]) if pd.notna(test_m["auc_ovr_weighted"]) else np.nan,
                "best_estimator": best_pipe,
                "test_pred": yhat,
                "test_proba": proba,
                "threshold": np.nan,
                "is_best_train": int(name == best_name),
            })

    df_out = pd.DataFrame([{k: v for k, v in row.items() if k not in ("best_estimator", "test_pred", "test_proba", "threshold")} for row in rows])
    return df_out, rows, best_name

def run_block(Xtr, Xte, ytr, yte, task, block_name, out_dir, title_prefix, stage_names=None):
    models_df, rows, best_name = fit_and_test_all_models(Xtr, ytr, Xte, yte, task)
    best_row = next(r for r in rows if r["model"] == best_name)

    vis_pre = Pipeline([("pre", SavGolSNV()), ("scaler", StandardScaler())])
    Xtr_vis = vis_pre.fit_transform(Xtr, ytr)
    Xte_vis = vis_pre.transform(Xte)

    pc1, pc2 = plot_pca_test_only(
        Xtr_vis, Xte_vis, yte,
        out_png_test=os.path.join(out_dir, f"{block_name}__PCA_test.png"),
        title_prefix=title_prefix
    )

    if task == "doenca":
        save_confusion_matrix(
            yte, best_row["test_pred"], labels=[0, 1], tick_labels=[0, 1],
            out_png=os.path.join(out_dir, f"{block_name}__CM_test.png"),
            title=f"TEST CM | {title_prefix} | {best_name}"
        )
        if best_row["test_proba"] is not None:
            plot_roc_binary(
                yte, best_row["test_proba"],
                os.path.join(out_dir, f"{block_name}__ROC_test.png"),
                f"ROC (TEST) | {title_prefix} | {best_name}"
            )
    else:
        labels = list(np.arange(len(stage_names)))
        save_confusion_matrix(
            yte, best_row["test_pred"], labels=labels, tick_labels=stage_names,
            out_png=os.path.join(out_dir, f"{block_name}__CM_test.png"),
            title=f"TEST CM | {title_prefix} | {best_name}"
        )
        if best_row["test_proba"] is not None:
            plot_roc_multiclass_ovr_macro(
                yte, best_row["test_proba"], len(stage_names),
                os.path.join(out_dir, f"{block_name}__ROC_test.png"),
                f"ROC macro OvR (TEST) | {title_prefix} | {best_name}"
            )

    return models_df, rows, best_name, pc1, pc2

def make_block_table(task_label, block_name, interval_text, df_models):
    rows = []
    for _, r in df_models.iterrows():
        rows.append({
            "Task": task_label,
            "Block": block_name,
            "Intervalo do espectro": interval_text,
            "Algoritmo": r["model"],
            "Train_score": r["Train_nestedCV_mean_score"],
            "Train_accuracy": r["Train_accuracy"],
            "Train_precision": r["Train_precision"],
            "Train_recall": r["Train_recall"],
            "Train_f1": r["Train_f1"],
            "Train_specificity": r["Train_specificity"],
            "Train_auc": r["Train_auc"],
            "Test_accuracy": r["Test_accuracy"],
            "Test_precision": r["Test_precision"],
            "Test_recall": r["Test_recall"],
            "Test_f1": r["Test_f1"],
            "Test_specificity": r["Test_specificity"],
            "Test_auc": r["Test_auc"],
        })
    return pd.DataFrame(rows)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_csv_robust(CSV_PATH)
    col_doenca = find_label_col(df, candidates=["doenca"])
    col_estadio = find_label_col(df, candidates=["estadio"])
    if col_doenca is None or col_estadio is None:
        raise ValueError("Could not find label columns for Doenca/Estadio.")

    spectral_cols, axis_vals = parse_spectral_columns(df)
    df[spectral_cols] = coerce_numeric(df, spectral_cols)
    df = df.dropna(subset=spectral_cols + [col_doenca, col_estadio]).copy()

    X_all = df[spectral_cols].to_numpy(dtype=float)
    y_doenca = df[col_doenca].astype(int).to_numpy()

    le_stage = LabelEncoder()
    y_stage = le_stage.fit_transform(df[col_estadio].astype(str).to_numpy())
    stage_names = list(le_stage.classes_)

    log(f"Eixo detetado: {axis_vals.min():.0f}–{axis_vals.max():.0f} | n_vars={len(spectral_cols)}")
    log(f"Classes Estádio: {stage_names}")

    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_all, y_doenca, test_size=TEST_SIZE, stratify=y_doenca, random_state=RANDOM_STATE
    )
    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_all, y_stage, test_size=TEST_SIZE, stratify=y_stage, random_state=RANDOM_STATE
    )

    log("[A] FULL SPECTRUM")
    doenca_full_df, doenca_full_rows, doenca_full_best, pc1_df, pc2_df = run_block(
        X_tr_d, X_te_d, y_tr_d, y_te_d, "doenca", "DOENCA__FULLSPEC", OUT_DIR, "Doenca | FULLSPEC"
    )
    estadio_full_df, estadio_full_rows, estadio_full_best, pc1_sf, pc2_sf = run_block(
        X_tr_s, X_te_s, y_tr_s, y_te_s, "estadio", "ESTADIO__FULLSPEC", OUT_DIR, "Estadio | FULLSPEC", stage_names=stage_names
    )

    log("[B] FULL SEARCH")
    rank_d_full = rank_windows(X_tr_d, axis_vals, y_tr_d, "doenca", "FULL_SEARCH").head(TOPK_WINDOWS_FULL)
    rank_s_full = rank_windows(X_tr_s, axis_vals, y_tr_s, "estadio", "FULL_SEARCH").head(TOPK_WINDOWS_FULL)
    plot_score_vs_windows(rank_d_full, "Doenca", "FULL_SEARCH", OUT_DIR)
    plot_score_vs_windows(rank_s_full, "Estadio", "FULL_SEARCH", OUT_DIR)

    best_win_d_full = (float(rank_d_full.loc[0, "start_cm-1"]), float(rank_d_full.loc[0, "end_cm-1"]))
    best_win_s_full = (float(rank_s_full.loc[0, "start_cm-1"]), float(rank_s_full.loc[0, "end_cm-1"]))

    Xtr_d_full_w, idx_d_full = slice_window_by_axis(X_tr_d, axis_vals, best_win_d_full[0], best_win_d_full[1])
    Xte_d_full_w, _ = slice_window_by_axis(X_te_d, axis_vals, best_win_d_full[0], best_win_d_full[1])
    Xtr_s_full_w, idx_s_full = slice_window_by_axis(X_tr_s, axis_vals, best_win_s_full[0], best_win_s_full[1])
    Xte_s_full_w, _ = slice_window_by_axis(X_te_s, axis_vals, best_win_s_full[0], best_win_s_full[1])

    doenca_bfull_df, doenca_bfull_rows, doenca_bfull_best, pc1_db, pc2_db = run_block(
        Xtr_d_full_w, Xte_d_full_w, y_tr_d, y_te_d, "doenca", "DOENCA__BESTWINDOW_FULL", OUT_DIR,
        f"Doenca | BESTWINDOW FULL | {best_win_d_full[0]:.0f}-{best_win_d_full[1]:.0f}"
    )
    estadio_bfull_df, estadio_bfull_rows, estadio_bfull_best, pc1_sb, pc2_sb = run_block(
        Xtr_s_full_w, Xte_s_full_w, y_tr_s, y_te_s, "estadio", "ESTADIO__BESTWINDOW_FULL", OUT_DIR,
        f"Estadio | BESTWINDOW FULL | {best_win_s_full[0]:.0f}-{best_win_s_full[1]:.0f}", stage_names=stage_names
    )

    best_row = next(r for r in doenca_bfull_rows if r["model"] == doenca_bfull_best)
    joblib.dump({
        "task": "nir_doenca_binaria",
        "block": "B_FULL_SEARCH",
        "pipeline": best_row["best_estimator"],
        "threshold": best_row["threshold"],
        "window_cm-1": {"start": best_win_d_full[0], "end": best_win_d_full[1]},
        "best_model_name": doenca_bfull_best,
        "pc_variance_test": {"PC1_pct": pc1_db, "PC2_pct": pc2_db},
    }, os.path.join(OUT_DIR, "DOENCA__BESTWINDOW_FULL__best_model.joblib"))

    best_row = next(r for r in estadio_bfull_rows if r["model"] == estadio_bfull_best)
    joblib.dump({
        "task": "nir_estadio_multiclasse",
        "block": "B_FULL_SEARCH",
        "pipeline": best_row["best_estimator"],
        "label_encoder": le_stage,
        "class_names": stage_names,
        "window_cm-1": {"start": best_win_s_full[0], "end": best_win_s_full[1]},
        "best_model_name": estadio_bfull_best,
        "pc_variance_test": {"PC1_pct": pc1_sb, "PC2_pct": pc2_sb},
    }, os.path.join(OUT_DIR, "ESTADIO__BESTWINDOW_FULL__best_model.joblib"))

    log("[C] miniNIR-like SEARCH")
    rank_d_mini = rank_windows(X_tr_d, axis_vals, y_tr_d, "doenca", "MININIR").head(TOPK_WINDOWS_MININIR)
    rank_s_mini = rank_windows(X_tr_s, axis_vals, y_tr_s, "estadio", "MININIR").head(TOPK_WINDOWS_MININIR)
    plot_score_vs_windows(rank_d_mini, "Doenca", "MININIR", OUT_DIR)
    plot_score_vs_windows(rank_s_mini, "Estadio", "MININIR", OUT_DIR)

    best_win_d_mini = (float(rank_d_mini.loc[0, "start_cm-1"]), float(rank_d_mini.loc[0, "end_cm-1"]))
    best_win_s_mini = (float(rank_s_mini.loc[0, "start_cm-1"]), float(rank_s_mini.loc[0, "end_cm-1"]))

    Xtr_d_mini_w, idx_d_mini = slice_window_by_axis(X_tr_d, axis_vals, best_win_d_mini[0], best_win_d_mini[1])
    Xte_d_mini_w, _ = slice_window_by_axis(X_te_d, axis_vals, best_win_d_mini[0], best_win_d_mini[1])
    Xtr_s_mini_w, idx_s_mini = slice_window_by_axis(X_tr_s, axis_vals, best_win_s_mini[0], best_win_s_mini[1])
    Xte_s_mini_w, _ = slice_window_by_axis(X_te_s, axis_vals, best_win_s_mini[0], best_win_s_mini[1])

    doenca_mini_df, doenca_mini_rows, doenca_mini_best, pc1_dm, pc2_dm = run_block(
        Xtr_d_mini_w, Xte_d_mini_w, y_tr_d, y_te_d, "doenca", "DOENCA__BESTWINDOW_MININIR", OUT_DIR,
        f"Doenca | BESTWINDOW MININIR | {best_win_d_mini[0]:.0f}-{best_win_d_mini[1]:.0f}"
    )
    estadio_mini_df, estadio_mini_rows, estadio_mini_best, pc1_sm, pc2_sm = run_block(
        Xtr_s_mini_w, Xte_s_mini_w, y_tr_s, y_te_s, "estadio", "ESTADIO__BESTWINDOW_MININIR", OUT_DIR,
        f"Estadio | BESTWINDOW MININIR | {best_win_s_mini[0]:.0f}-{best_win_s_mini[1]:.0f}", stage_names=stage_names
    )

    best_row = next(r for r in doenca_mini_rows if r["model"] == doenca_mini_best)
    joblib.dump({
        "task": "nir_doenca_binaria",
        "block": "C_MININIR",
        "pipeline": best_row["best_estimator"],
        "threshold": best_row["threshold"],
        "window_cm-1": {"start": best_win_d_mini[0], "end": best_win_d_mini[1]},
        "best_model_name": doenca_mini_best,
        "pc_variance_test": {"PC1_pct": pc1_dm, "PC2_pct": pc2_dm},
    }, os.path.join(OUT_DIR, "DOENCA__BESTWINDOW_MININIR__best_model.joblib"))

    best_row = next(r for r in estadio_mini_rows if r["model"] == estadio_mini_best)
    joblib.dump({
        "task": "nir_estadio_multiclasse",
        "block": "C_MININIR",
        "pipeline": best_row["best_estimator"],
        "label_encoder": le_stage,
        "class_names": stage_names,
        "window_cm-1": {"start": best_win_s_mini[0], "end": best_win_s_mini[1]},
        "best_model_name": estadio_mini_best,
        "pc_variance_test": {"PC1_pct": pc1_sm, "PC2_pct": pc2_sm},
    }, os.path.join(OUT_DIR, "ESTADIO__BESTWINDOW_MININIR__best_model.joblib"))

    final_xlsx = os.path.join(OUT_DIR, "NIR__CLASSIFICATION__LITE_V2_OUTPUT.xlsx")
    with pd.ExcelWriter(final_xlsx, engine="openpyxl") as xw:
        rank_d_full.to_excel(xw, index=False, sheet_name="WinRank_Doenca_FULL")
        rank_s_full.to_excel(xw, index=False, sheet_name="WinRank_Estadio_FULL")
        rank_d_mini.to_excel(xw, index=False, sheet_name="WinRank_Doenca_MINI")
        rank_s_mini.to_excel(xw, index=False, sheet_name="WinRank_Estadio_MINI")

        doenca_full_df.to_excel(xw, index=False, sheet_name="Models_Doenca_FULLSPEC")
        estadio_full_df.to_excel(xw, index=False, sheet_name="Models_Estadio_FULLSPEC")
        doenca_bfull_df.to_excel(xw, index=False, sheet_name="Models_Doenca_BESTFULL")
        estadio_bfull_df.to_excel(xw, index=False, sheet_name="Models_Estadio_BESTFULL")
        doenca_mini_df.to_excel(xw, index=False, sheet_name="Models_Doenca_MININIR")
        estadio_mini_df.to_excel(xw, index=False, sheet_name="Models_Estadio_MININIR")

        template = pd.concat([
            make_block_table("Doenca", "A_FULLSPEC", "ALL", doenca_full_df),
            make_block_table("Estadio", "A_FULLSPEC", "ALL", estadio_full_df),
            make_block_table("Doenca", "B_BESTWINDOW_FULL", f"{best_win_d_full[0]:.0f}-{best_win_d_full[1]:.0f}", doenca_bfull_df),
            make_block_table("Estadio", "B_BESTWINDOW_FULL", f"{best_win_s_full[0]:.0f}-{best_win_s_full[1]:.0f}", estadio_bfull_df),
            make_block_table("Doenca", "C_BESTWINDOW_MININIR", f"{best_win_d_mini[0]:.0f}-{best_win_d_mini[1]:.0f}", doenca_mini_df),
            make_block_table("Estadio", "C_BESTWINDOW_MININIR", f"{best_win_s_mini[0]:.0f}-{best_win_s_mini[1]:.0f}", estadio_mini_df),
        ], ignore_index=True)
        template.to_excel(xw, index=False, sheet_name="Tabela_formato_template")

        pd.DataFrame([{
            "Doenca_A_best": doenca_full_best,
            "Estadio_A_best": estadio_full_best,
            "Doenca_B_window": f"{best_win_d_full[0]:.0f}-{best_win_d_full[1]:.0f}",
            "Doenca_B_best": doenca_bfull_best,
            "Estadio_B_window": f"{best_win_s_full[0]:.0f}-{best_win_s_full[1]:.0f}",
            "Estadio_B_best": estadio_bfull_best,
            "Doenca_C_window": f"{best_win_d_mini[0]:.0f}-{best_win_d_mini[1]:.0f}",
            "Doenca_C_best": doenca_mini_best,
            "Estadio_C_window": f"{best_win_s_mini[0]:.0f}-{best_win_s_mini[1]:.0f}",
            "Estadio_C_best": estadio_mini_best,
        }]).to_excel(xw, index=False, sheet_name="SUMMARY")

    zip_path = shutil.make_archive(OUT_DIR, "zip", OUT_DIR)

    log("OK ✅")
    log(f"Outputs folder: {OUT_DIR}")
    log(f"Excel: {final_xlsx}")
    log(f"ZIP: {zip_path}")

if __name__ == "__main__":
    main()
