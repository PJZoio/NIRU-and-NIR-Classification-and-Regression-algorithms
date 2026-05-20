# -*- coding: utf-8 -*-
"""
NIR regression - LITE v3
For ONE analyte at a time, this script runs:
(a) FULL SPECTRUM
(b) FULL-SPECTRUM WINDOW SEARCH
(c) miniNIR-like WINDOW SEARCH

All blocks report TRAIN and TEST for ALL algorithms.
Best model plots are generated for (a), (b), (c).
Best models are saved for (b) and (c).
"""

import os
import re
import shutil
import unicodedata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

CSV_PATH = "Compostas (Novas) - Global.csv"
OUT_DIR = "NIR_regression_outputs_lite_v3"

ANALYTE = "Creatinina"   # Albumina / Creatinina / Ureia

OUTER_SPLITS = 3
INNER_SPLITS = 2

TOPK_WINDOWS_FULL = 3
TOPK_WINDOWS_MININIR = 3

WIN_WIDTH_FULL = 600
STEP_FULL = 300
MIN_VARS_FULL = 100

MININIR_HALF_WIDTH = 250.0
MIN_VARS_MININIR = 20

TEST_SIZE = 0.20
RANDOM_STATE = 42
VERBOSE = True


def log(msg: str):
    if VERBOSE:
        print(msg, flush=True)


def read_csv_robust(path):
    p = Path(path)
    if p.is_dir():
        csvs = list(p.glob("*.csv")) or list(p.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV found in directory: {path}")
        p = csvs[0]
        log(f"Using CSV file: {p}")
    for enc in ["utf-8", "cp1252", "latin-1"]:
        try:
            return pd.read_csv(p, sep=";", decimal=",", encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.read_csv(p, sep=";", decimal=",", low_memory=False)


def clean(s: str) -> str:
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s)
    return s


def find_target_col(df, keywords_any, keywords_all=None):
    cleaned = {clean(c): c for c in df.columns}
    for ck, orig in cleaned.items():
        if keywords_all and not all(k in ck for k in keywords_all):
            continue
        if any(k in ck for k in keywords_any):
            return orig
    return None


def parse_spectral_columns(df, exclude=None):
    if exclude is None:
        exclude = {
            "doenca", "estadio", "type", "conc", "conc albumina", "conc creatinina",
            "conc ureia", "conc urea", "albumina", "creatinina", "ureia", "urea", "unnamed: 0"
        }
    cols, vals = [], []
    for c in df.columns:
        if clean(c) in exclude:
            continue
        s = str(c).strip().replace(",", ".")
        try:
            v = float(s)
            cols.append(c); vals.append(v)
        except Exception:
            m = re.search(r"([0-9]+(?:[.,][0-9]+)?)", str(c))
            if m:
                v = float(m.group(1).replace(",", "."))
                cols.append(c); vals.append(v)
    idx = np.argsort(vals)
    cols_sorted = [cols[i] for i in idx]
    vals_sorted = np.array([vals[i] for i in idx], dtype=float)
    return cols_sorted, vals_sorted


def coerce_numeric_matrix(df, cols):
    out = df[cols].copy()
    for c in cols:
        if out[c].dtype.kind in "biufc":
            continue
        s = out[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        s = s.str.replace(r"[^0-9\.\-eE+]", "", regex=True)
        out[c] = pd.to_numeric(s, errors="coerce")
    return out


def slice_window_idx(axis_vals, start, end):
    lo, hi = (start, end) if start <= end else (end, start)
    mask = (axis_vals >= lo) & (axis_vals <= hi)
    return np.where(mask)[0]


def nm_to_cm1(nm):
    return 1e7 / float(nm)


def generate_full_windows(axis_vals):
    windows = []
    lo, hi = float(axis_vals.min()), float(axis_vals.max())
    start = lo
    while start + WIN_WIDTH_FULL <= hi:
        end = start + WIN_WIDTH_FULL
        idx = slice_window_idx(axis_vals, start, end)
        if len(idx) >= MIN_VARS_FULL:
            windows.append((float(start), float(end), idx))
        start += STEP_FULL
    return windows


def generate_mininir_like_windows(axis_vals):
    windows = []
    centers = [nm_to_cm1(x) for x in [1300, 1400, 1600, 1700, 1900, 2100]]
    for c in centers:
        idx = slice_window_idx(axis_vals, c - MININIR_HALF_WIDTH, c + MININIR_HALF_WIDTH)
        if len(idx) >= MIN_VARS_MININIR:
            windows.append((float(c - MININIR_HALF_WIDTH), float(c + MININIR_HALF_WIDTH), idx))
    merged = [
        (nm_to_cm1(1400) - MININIR_HALF_WIDTH, nm_to_cm1(1300) + MININIR_HALF_WIDTH),
        (nm_to_cm1(1700) - MININIR_HALF_WIDTH, nm_to_cm1(1600) + MININIR_HALF_WIDTH),
        (nm_to_cm1(2100) - MININIR_HALF_WIDTH, nm_to_cm1(1900) + MININIR_HALF_WIDTH),
    ]
    for s, e in merged:
        idx = slice_window_idx(axis_vals, s, e)
        if len(idx) >= MIN_VARS_MININIR:
            windows.append((float(min(s, e)), float(max(s, e)), idx))
    uniq = []
    seen = set()
    for s, e, idx in windows:
        key = (round(s, 3), round(e, 3), len(idx))
        if key not in seen:
            uniq.append((s, e, idx))
            seen.add(key)
    return uniq


class SavGolSNV(BaseEstimator, TransformerMixin):
    def __init__(self, window=11, poly=2, deriv=1):
        self.window = window
        self.poly = poly
        self.deriv = deriv

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
            Xf = savgol_filter(X, window_length=w, polyorder=min(self.poly, w - 1),
                               deriv=self.deriv, axis=1)
        mu = Xf.mean(axis=1, keepdims=True)
        sd = Xf.std(axis=1, keepdims=True) + 1e-12
        return (Xf - mu) / sd


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def reg_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def model_grids_lite():
    return {
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"model__alpha": [0.1, 1.0, 10.0]}
        ),
        "PLS": (
            PLSRegression(),
            {"model__n_components": [2, 4, 6]}
        ),
        "SVR_RBF": (
            SVR(kernel="rbf"),
            {"model__C": [1, 10], "model__gamma": ["scale", 0.01], "model__epsilon": [0.01, 0.1]}
        ),
        "RF": (
            RandomForestRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [300], "model__max_depth": [8, None],
             "model__min_samples_leaf": [2, 5], "model__max_features": ["sqrt"]}
        ),
        "GB": (
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            {"model__n_estimators": [300], "model__learning_rate": [0.05, 0.1],
             "model__max_depth": [2, 3], "model__subsample": [0.7]}
        )
    }


def screen_windows_ridge(X, y, windows, cv_splits=3, tag="FULL"):
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    pipe = Pipeline([
        ("pre", SavGolSNV(window=11, poly=2, deriv=1)),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=RANDOM_STATE))
    ])
    total = len(windows)
    for i, (s, e, idx) in enumerate(windows, start=1):
        log(f"[SCREEN {tag}] Window {i}/{total}: {s:.0f}-{e:.0f} cm-1 | n={len(idx)}")
        Xw = X[:, idx]
        r2s = []
        for tr, te in cv.split(Xw):
            pipe.fit(Xw[tr], y[tr])
            pred = pipe.predict(Xw[te])
            r2s.append(r2_score(y[te], pred))
        rows.append({
            "start_cm-1": float(s),
            "end_cm-1": float(e),
            "window_label": f"{s:.0f}-{e:.0f}",
            "n_features": int(len(idx)),
            "R2_mean_screen": float(np.mean(r2s)),
            "R2_std_screen": float(np.std(r2s)),
        })
    return pd.DataFrame(rows).sort_values("R2_mean_screen", ascending=False).reset_index(drop=True)


def evaluate_all_models_nested_lite(Xw, y, outer_splits=3, inner_splits=2):
    outer = KFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    model_rows = []
    best_name = None
    best_outer_r2 = -np.inf
    grids = model_grids_lite()

    for model_name, (mdl, grid) in grids.items():
        log(f"    [MODEL] {model_name}")
        outer_r2s, outer_rmses, outer_maes = [], [], []
        for tr, te in outer.split(Xw):
            Xtr, Xte = Xw[tr], Xw[te]
            ytr, yte = y[tr], y[te]
            pipe = Pipeline([
                ("pre", SavGolSNV(window=11, poly=2, deriv=1)),
                ("scaler", StandardScaler()),
                ("model", mdl)
            ])
            inner = KFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)
            gs = GridSearchCV(pipe, grid, scoring="r2", cv=inner, n_jobs=-1)
            gs.fit(Xtr, ytr)
            pred = gs.best_estimator_.predict(Xte)
            m = reg_metrics(yte, pred)
            outer_r2s.append(m["R2"])
            outer_rmses.append(m["RMSE"])
            outer_maes.append(m["MAE"])
        row = {
            "model": model_name,
            "Train_R2": float(np.mean(outer_r2s)),
            "Train_RMSE": float(np.mean(outer_rmses)),
            "Train_MAE": float(np.mean(outer_maes)),
        }
        model_rows.append(row)
        if row["Train_R2"] > best_outer_r2:
            best_outer_r2 = row["Train_R2"]
            best_name = model_name

    return pd.DataFrame(model_rows).sort_values("Train_R2", ascending=False).reset_index(drop=True), best_name


def fit_and_test_all_models(Xtr, ytr, Xte, yte):
    models_df, best_name = evaluate_all_models_nested_lite(Xtr, ytr, OUTER_SPLITS, INNER_SPLITS)
    rows = []
    for _, r in models_df.iterrows():
        name = r["model"]
        mdl, grid = model_grids_lite()[name]
        pipe = Pipeline([
            ("pre", SavGolSNV(window=11, poly=2, deriv=1)),
            ("scaler", StandardScaler()),
            ("model", mdl)
        ])
        inner = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, grid, scoring="r2", cv=inner, n_jobs=-1)
        gs.fit(Xtr, ytr)
        pred_te = np.asarray(gs.best_estimator_.predict(Xte)).ravel()
        test_m = reg_metrics(yte, pred_te)
        rows.append({
            "model": name,
            "Train_R2": float(r["Train_R2"]),
            "Train_RMSE": float(r["Train_RMSE"]),
            "Train_MAE": float(r["Train_MAE"]),
            "Test_R2": float(test_m["R2"]),
            "Test_RMSE": float(test_m["RMSE"]),
            "Test_MAE": float(test_m["MAE"]),
            "best_estimator": gs.best_estimator_,
            "test_pred": pred_te,
            "is_best_train": int(name == best_name),
        })
    df_out = pd.DataFrame([{k: v for k, v in row.items() if k not in ("best_estimator", "test_pred")} for row in rows])
    return df_out, rows, best_name


def pca_plot_test_only(Xtr_proc, Xte_proc, ytr, yte, analyte, out_dir, suffix=""):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca.fit(Xtr_proc)
    Zte = pca.transform(Xte_proc)
    pc1 = pca.explained_variance_ratio_[0] * 100
    pc2 = pca.explained_variance_ratio_[1] * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(Zte[:, 0], Zte[:, 1], c=yte)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(f"{analyte} concentration")
    ax.set_xlabel(f"PC1 ({pc1:.1f}%)")
    ax.set_ylabel(f"PC2 ({pc2:.1f}%)")
    ax.set_title(f"{analyte} | PCA test {suffix}".strip())
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"PCA_{analyte}__test{suffix}.png", dpi=200)
    plt.close(fig)
    return pc1, pc2


def pred_vs_actual_plot(y_true, y_pred, analyte, out_dir, suffix=""):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred)
    lo = min(float(np.min(y_true)), float(np.min(y_pred)))
    hi = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi])
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{analyte} | Predicted vs Actual (TEST) {suffix}".strip())
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"Pred_vs_Actual_{analyte}__test{suffix}.png", dpi=200)
    plt.close(fig)


def r2_vs_windows_plot(screen_df, analyte, tag, out_dir):
    labels = screen_df["window_label"].tolist()
    y = screen_df["R2_mean_screen"].to_numpy()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, marker="o")
    N = max(1, len(labels) // 12)
    ax.set_xticks(x[::N])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), N)], rotation=45, ha="right")
    ax.set_xlabel("Spectral window (cm$^{-1}$)")
    ax.set_ylabel("R$^2$ (CV mean)")
    ax.set_title(f"{analyte} | R$^2$ across candidate windows | {tag}")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"R2_vs_windows__{analyte}__{tag}.png", dpi=200)
    plt.close(fig)


def block_df(analyte, block_name, interval_text, models_df):
    return pd.DataFrame([{
        "Analyte": analyte,
        "Block": block_name,
        "Intervalo do espectro": interval_text,
        "Algoritmo": row["model"],
        "Train_R2": row["Train_R2"],
        "Train_RMSE": row["Train_RMSE"],
        "Train_MAE": row["Train_MAE"],
        "Test_R2": row["Test_R2"],
        "Test_RMSE": row["Test_RMSE"],
        "Test_MAE": row["Test_MAE"],
    } for _, row in models_df.iterrows()])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = read_csv_robust(CSV_PATH)

    col_albumina = find_target_col(df, keywords_any=["albumina"], keywords_all=["conc"])
    col_creatinina = find_target_col(df, keywords_any=["creatinina"], keywords_all=["conc"])
    col_ureia = find_target_col(df, keywords_any=["ureia", "urea"], keywords_all=["conc"])

    target_map = {
        "Albumina": col_albumina,
        "Creatinina": col_creatinina,
        "Ureia": col_ureia,
    }
    if ANALYTE not in target_map or target_map[ANALYTE] is None:
        raise ValueError(f"Target column not found for analyte={ANALYTE}. Detected: {target_map}")

    spectral_cols, axis_vals = parse_spectral_columns(df)
    df[spectral_cols] = coerce_numeric_matrix(df, spectral_cols)

    use_cols = spectral_cols + [target_map[ANALYTE]]
    dfx = df[use_cols].copy()
    dfx[target_map[ANALYTE]] = pd.to_numeric(dfx[target_map[ANALYTE]], errors="coerce")
    dfx = dfx.dropna()

    X = dfx[spectral_cols].to_numpy(dtype=float)
    y = dfx[target_map[ANALYTE]].to_numpy(dtype=float)

    log(f"Analyte: {ANALYTE}")
    log(f"Samples: {len(y)} | Spectral vars: {X.shape[1]}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    vis = Pipeline([("pre", SavGolSNV(window=11, poly=2, deriv=1)), ("scaler", StandardScaler())])

    log("[A] FULL SPECTRUM")
    full_models_df, full_rows, full_best_name = fit_and_test_all_models(X_tr, y_tr, X_te, y_te)
    full_best_row = next(r for r in full_rows if r["model"] == full_best_name)

    Xtr_full_vis = vis.fit_transform(X_tr, y_tr)
    Xte_full_vis = vis.transform(X_te)
    pc1_a, pc2_a = pca_plot_test_only(Xtr_full_vis, Xte_full_vis, y_tr, y_te, ANALYTE, OUT_DIR, suffix="__A_FULL")
    pred_vs_actual_plot(y_te, full_best_row["test_pred"], ANALYTE, OUT_DIR, suffix="__A_FULL")

    log("[B] FULL-SPECTRUM WINDOW SEARCH")
    windows_full = generate_full_windows(axis_vals)
    rank_full = screen_windows_ridge(X_tr, y_tr, windows_full, cv_splits=3, tag="FULL_SEARCH")
    r2_vs_windows_plot(rank_full, ANALYTE, "FULL_SEARCH", OUT_DIR)

    topk_full = min(TOPK_WINDOWS_FULL, len(rank_full))
    best_b = None
    best_b_models_df = None
    best_b_rows = None
    best_b_name = None

    for i in range(topk_full):
        s = float(rank_full.loc[i, "start_cm-1"])
        e = float(rank_full.loc[i, "end_cm-1"])
        idx = slice_window_idx(axis_vals, s, e)
        log(f"[B TOP {i+1}/{topk_full}] {s:.0f}-{e:.0f} cm-1 | n={len(idx)}")
        Xtr_w = X_tr[:, idx]
        Xte_w = X_te[:, idx]
        models_df, rows, best_name = fit_and_test_all_models(Xtr_w, y_tr, Xte_w, y_te)
        score_train = float(models_df.loc[models_df["model"] == best_name, "Train_R2"].iloc[0])
        if (best_b is None) or (score_train > best_b["score_train"]):
            best_b = {"start": s, "end": e, "idx": idx, "score_train": score_train, "Xtr": Xtr_w, "Xte": Xte_w}
            best_b_models_df = models_df.copy()
            best_b_rows = rows
            best_b_name = best_name

    best_b_label = f"{best_b['start']:.0f}-{best_b['end']:.0f}"
    best_b_row = next(r for r in best_b_rows if r["model"] == best_b_name)

    Xtr_b_vis = vis.fit_transform(best_b["Xtr"], y_tr)
    Xte_b_vis = vis.transform(best_b["Xte"])
    pc1_b, pc2_b = pca_plot_test_only(Xtr_b_vis, Xte_b_vis, y_tr, y_te, ANALYTE, OUT_DIR, suffix="__B_FULLSEARCH")
    pred_vs_actual_plot(y_te, best_b_row["test_pred"], ANALYTE, OUT_DIR, suffix="__B_FULLSEARCH")

    joblib.dump({
        "task": "nir_regression_lite_v3",
        "analyte": ANALYTE,
        "block": "B_FULL_SEARCH",
        "pipeline": best_b_row["best_estimator"],
        "window_cm-1": {"start": best_b["start"], "end": best_b["end"]},
        "spectral_idx": best_b["idx"].tolist(),
        "pc_var": {"PC1_pct": pc1_b, "PC2_pct": pc2_b},
    }, Path(OUT_DIR) / f"NIR_{ANALYTE}__B_FULLSEARCH__best_model.joblib")

    log("[C] miniNIR-like WINDOW SEARCH")
    windows_mini = generate_mininir_like_windows(axis_vals)
    rank_mini = screen_windows_ridge(X_tr, y_tr, windows_mini, cv_splits=3, tag="MININIR_LIKE")

    topk_mini = min(TOPK_WINDOWS_MININIR, len(rank_mini))
    best_c = None
    best_c_models_df = None
    best_c_rows = None
    best_c_name = None

    for i in range(topk_mini):
        s = float(rank_mini.loc[i, "start_cm-1"])
        e = float(rank_mini.loc[i, "end_cm-1"])
        idx = slice_window_idx(axis_vals, s, e)
        log(f"[C TOP {i+1}/{topk_mini}] {s:.0f}-{e:.0f} cm-1 | n={len(idx)}")
        Xtr_w = X_tr[:, idx]
        Xte_w = X_te[:, idx]
        models_df, rows, best_name = fit_and_test_all_models(Xtr_w, y_tr, Xte_w, y_te)
        score_train = float(models_df.loc[models_df["model"] == best_name, "Train_R2"].iloc[0])
        if (best_c is None) or (score_train > best_c["score_train"]):
            best_c = {"start": s, "end": e, "idx": idx, "score_train": score_train, "Xtr": Xtr_w, "Xte": Xte_w}
            best_c_models_df = models_df.copy()
            best_c_rows = rows
            best_c_name = best_name

    best_c_label = f"{best_c['start']:.0f}-{best_c['end']:.0f}"
    best_c_row = next(r for r in best_c_rows if r["model"] == best_c_name)

    Xtr_c_vis = vis.fit_transform(best_c["Xtr"], y_tr)
    Xte_c_vis = vis.transform(best_c["Xte"])
    pc1_c, pc2_c = pca_plot_test_only(Xtr_c_vis, Xte_c_vis, y_tr, y_te, ANALYTE, OUT_DIR, suffix="__C_MININIR")
    pred_vs_actual_plot(y_te, best_c_row["test_pred"], ANALYTE, OUT_DIR, suffix="__C_MININIR")

    joblib.dump({
        "task": "nir_regression_lite_v3",
        "analyte": ANALYTE,
        "block": "C_MININIR_LIKE",
        "pipeline": best_c_row["best_estimator"],
        "window_cm-1": {"start": best_c["start"], "end": best_c["end"]},
        "spectral_idx": best_c["idx"].tolist(),
        "pc_var": {"PC1_pct": pc1_c, "PC2_pct": pc2_c},
    }, Path(OUT_DIR) / f"NIR_{ANALYTE}__C_MININIR__best_model.joblib")

    final_xlsx = Path(OUT_DIR) / "NIR__REGRESSION__LITE_V3_FINAL.xlsx"
    with pd.ExcelWriter(final_xlsx, engine="openpyxl") as writer:
        rank_full.to_excel(writer, sheet_name="WinRank_FULL", index=False)
        rank_mini.to_excel(writer, sheet_name="WinRank_MININIR", index=False)

        full_models_df.to_excel(writer, sheet_name="Models_FULLSPECTRUM", index=False)
        best_b_models_df.to_excel(writer, sheet_name="Models_BESTWINDOW_FULL", index=False)
        best_c_models_df.to_excel(writer, sheet_name="Models_BESTWINDOW_MININIR", index=False)

        template = pd.concat([
            block_df(ANALYTE, "A_FULL_SPECTRUM", "ALL", full_models_df),
            block_df(ANALYTE, "B_BEST_WINDOW_FULL", best_b_label, best_b_models_df),
            block_df(ANALYTE, "C_BEST_WINDOW_MININIR", best_c_label, best_c_models_df),
        ], ignore_index=True)
        template.to_excel(writer, sheet_name="Tabela_formato_template", index=False)

        pd.DataFrame([{
            "Analyte": ANALYTE,
            "A_best_model": full_best_name,
            "A_best_test_R2": float(full_models_df.loc[full_models_df["model"] == full_best_name, "Test_R2"].iloc[0]),
            "B_best_window": best_b_label,
            "B_best_model": best_b_name,
            "B_best_test_R2": float(best_b_models_df.loc[best_b_models_df["model"] == best_b_name, "Test_R2"].iloc[0]),
            "C_best_window": best_c_label,
            "C_best_model": best_c_name,
            "C_best_test_R2": float(best_c_models_df.loc[best_c_models_df["model"] == best_c_name, "Test_R2"].iloc[0]),
            "PC1_A_pct": pc1_a, "PC2_A_pct": pc2_a,
            "PC1_B_pct": pc1_b, "PC2_B_pct": pc2_b,
            "PC1_C_pct": pc1_c, "PC2_C_pct": pc2_c,
        }]).to_excel(writer, sheet_name="SUMMARY", index=False)

    zip_path = shutil.make_archive(OUT_DIR, "zip", OUT_DIR)

    log("Done ✅")
    log(f"Analyte: {ANALYTE}")
    log(f"A best model: {full_best_name}")
    log(f"B best window/model: {best_b_label} / {best_b_name}")
    log(f"C best window/model: {best_c_label} / {best_c_name}")
    log(f"Outputs: {OUT_DIR}")
    log(f"ZIP: {zip_path}")


if __name__ == "__main__":
    main()
