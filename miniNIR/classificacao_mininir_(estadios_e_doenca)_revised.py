
# -*- coding: utf-8 -*-
"""
Classificação miniNIR (Doença e Estádios) — versão revista

Principais melhorias:
1) Exporta métricas de treino/CV e de validação/teste para TODOS os modelos comparados.
2) Mantém apenas o MELHOR modelo guardado em joblib.
3) Gera visualizações do melhor modelo:
   - PCA (train)
   - Matriz de confusão (test)
   - ROC (Doença: binária; Estádios: multiclass one-vs-rest)
4) Gera ranking de features (LEDs) por permutation importance para o melhor modelo,
   com p-values de associação univariada:
   - Doença: Welch t-test
   - Estádios: ANOVA one-way
5) Gráficos com estilo limpo, apropriado para manuscrito.
"""

import os
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.signal import savgol_filter
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Colab opcional
try:
    from google.colab import files
    IN_COLAB = True
except Exception:
    IN_COLAB = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------
# Estilo de figuras (limpo / tipo journal)
# -------------------------
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.5,
})

# -------------------------
# SavGol + SNV por amostra
# -------------------------
class SavGolSNV(BaseEstimator, TransformerMixin):
    def __init__(self, window=5, poly=2, deriv=0):
        self.window = window
        self.poly = poly
        self.deriv = deriv

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        n_feat = X.shape[1]

        w = self.window
        if w % 2 == 0:
            w += 1
        if w > n_feat:
            w = n_feat if n_feat % 2 == 1 else max(3, n_feat - 1)
        if w < 3:
            Xf = X.copy()
        else:
            Xf = savgol_filter(
                X, window_length=w,
                polyorder=min(self.poly, w - 1),
                deriv=self.deriv, axis=1
            )

        mu = Xf.mean(axis=1, keepdims=True)
        sd = Xf.std(axis=1, keepdims=True) + 1e-12
        return (Xf - mu) / sd

# -------------------------
# Funções de plotting
# -------------------------
def plot_pca(X_proc, y, out_png, title, class_names=None):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X_proc)

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    classes = np.unique(y)
    for i, c in enumerate(classes):
        idx = (y == c)
        label = class_names[c] if (class_names is not None and isinstance(c, (int, np.integer)) and c < len(class_names)) else str(c)
        ax.scatter(Z[idx, 0], Z[idx, 1], label=label, alpha=0.85, s=28, edgecolor="black", linewidth=0.25)

    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.grid(True)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left", title="Class")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def save_confusion(y_true, y_pred, labels, out_png, title, class_names=None):
    display_labels = class_names if class_names is not None else labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=display_labels)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    disp.plot(ax=ax, values_format="d", cmap="Greys", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_roc_binary(y_true, y_proba, out_png, title):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def plot_roc_multiclass_ovr(y_true, y_proba, class_names, out_png, title):
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(6.0, 4.8))

    aucs = []
    for i, cname in enumerate(class_names):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.3f})")

    # macro-average simple
    macro_auc = np.mean(aucs) if len(aucs) else np.nan
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"{title} | macro AUC={macro_auc:.3f}" if len(aucs) else title)
    ax.grid(True)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# -------------------------
# Métricas
# -------------------------
def specificity_binary(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan

def metrics_binary(y_true, y_pred, y_proba):
    out = {}
    out["roc_auc"] = roc_auc_score(y_true, y_proba) if y_proba is not None else np.nan
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision"] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    out["recall"] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)  # sensitivity
    out["f1_score"] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    out["specificity"] = specificity_binary(y_true, y_pred)
    return out

def metrics_multiclass(y_true, y_pred, y_proba=None):
    out = {}
    out["roc_auc_ovr_weighted"] = (
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        if y_proba is not None else np.nan
    )
    out["accuracy"] = accuracy_score(y_true, y_pred)
    out["precision_macro"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    out["recall_macro"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    out["f1_macro"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    out["f1_weighted"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return out

# -------------------------
# Model grids
# -------------------------
def model_grids_binary():
    grids = {}
    grids["SVM_RBF"] = (
        SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.1, 0.01]}
    )
    grids["RF"] = (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 5, 8],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__max_features": ["sqrt", 0.5],
        }
    )
    if HAS_XGB:
        grids["XGBoost"] = (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_estimators=600,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=4,
                reg_lambda=1.0,
            ),
            {
                "clf__max_depth": [3, 4, 5],
                "clf__min_child_weight": [1, 5],
                "clf__subsample": [0.7, 0.9],
                "clf__colsample_bytree": [0.7, 0.9],
                "clf__reg_lambda": [0.5, 1.0, 2.0],
            }
        )
    return grids

def model_grids_multiclass():
    return model_grids_binary()

# -------------------------
# Threshold tuning (binário)
# -------------------------
def tune_threshold_for_recall(y_true, proba_pos, min_precision=None):
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t, best_recall, best_prec = 0.5, -np.inf, np.nan
    for t in thresholds:
        yhat = (proba_pos >= t).astype(int)
        rec = recall_score(y_true, yhat, pos_label=1, zero_division=0)
        prec = precision_score(y_true, yhat, pos_label=1, zero_division=0)
        if (min_precision is not None) and (prec < min_precision):
            continue
        if rec > best_recall:
            best_recall = rec
            best_t = t
            best_prec = prec
    return best_t, best_recall, best_prec

# -------------------------
# Ranking de features
# -------------------------
def transformed_feature_names(feature_cols, deriv=0):
    suf = "_d1" if deriv == 1 else ("_d2" if deriv == 2 else "")
    return [f"{c}{suf}" for c in feature_cols]

def feature_significance_binary(X_proc, y, feat_names):
    rows = []
    y = np.asarray(y)
    for j, f in enumerate(feat_names):
        g0 = X_proc[y == 0, j]
        g1 = X_proc[y == 1, j]
        stat, p = stats.ttest_ind(g0, g1, equal_var=False, nan_policy="omit")
        rows.append({"feature": f, "p_value": p})
    return pd.DataFrame(rows)

def feature_significance_multiclass(X_proc, y, feat_names):
    rows = []
    y = np.asarray(y)
    classes = np.unique(y)
    for j, f in enumerate(feat_names):
        groups = [X_proc[y == c, j] for c in classes]
        stat, p = stats.f_oneway(*groups)
        rows.append({"feature": f, "p_value": p})
    return pd.DataFrame(rows)

def permutation_importance_df(estimator, X, y, feat_names, scoring, n_repeats=200):
    r = permutation_importance(
        estimator, X, y, n_repeats=n_repeats, random_state=RANDOM_STATE,
        scoring=scoring, n_jobs=-1
    )
    return pd.DataFrame({
        "feature": feat_names,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)

def plot_feature_ranking(df_rank, out_png, title):
    d = df_rank.sort_values("importance_mean", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    ax.barh(d["feature"], d["importance_mean"], xerr=d["importance_std"], color="lightgray", edgecolor="black")
    for i, (_, row) in enumerate(d.iterrows()):
        ptxt = f"p={row['p_value']:.3g}" if pd.notna(row["p_value"]) else "p=NA"
        ax.text(
            row["importance_mean"] + max(d["importance_mean"].max(), 1e-6) * 0.03,
            i,
            ptxt,
            va="center",
            fontsize=7
        )
    ax.set_xlabel("Permutation importance")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

# =========================
# LOAD DATA
# =========================
DEFAULT_CSV = "2026.02.16-miniNir (solucoes compostas)-Doenca e Estadio.csv"

if IN_COLAB:
    uploaded = files.upload()
    # se DEFAULT_CSV não existir, tenta usar o primeiro csv encontrado
    if not os.path.exists(DEFAULT_CSV):
        csvs = [k for k in uploaded.keys() if k.lower().endswith(".csv")]
        if len(csvs) == 0:
            raise FileNotFoundError("Nenhum CSV carregado. Faz upload do ficheiro com os dados miniNIR.")
        DATA_FILE = csvs[0]
    else:
        DATA_FILE = DEFAULT_CSV
else:
    DATA_FILE = DEFAULT_CSV

df = pd.read_csv(DATA_FILE, sep=";", decimal=",")

feat_cols = [f"LED{i}_avg" for i in range(1, 7)]
missing = [c for c in feat_cols + ["Doenca", "Estadio"] if c not in df.columns]
if missing:
    raise ValueError(f"Colunas em falta no CSV: {missing}")

X = df[feat_cols].to_numpy()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# =========================
# TASK 1: DOENÇA (BINÁRIO)
# =========================
y_doenca = df["Doenca"].astype(int).to_numpy()

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X, y_doenca, test_size=0.2, stratify=y_doenca, random_state=RANDOM_STATE
)

rows_doenca_cv = []
rows_doenca_test = []
trained_binary_estimators = {}

best_doenca_name, best_doenca_score = None, -np.inf
best_doenca_estimator, best_thr = None, 0.5

for name, (clf, grid) in model_grids_binary().items():
    pipe = Pipeline([
        ("pre", SavGolSNV(window=5, poly=2, deriv=0)),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    gs = GridSearchCV(pipe, grid, scoring="recall", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_tr_d, y_tr_d)
    est = gs.best_estimator_
    trained_binary_estimators[name] = est

    # CV
    y_pred_cv = cross_val_predict(est, X_tr_d, y_tr_d, cv=cv, method="predict", n_jobs=-1)
    try:
        proba_cv = cross_val_predict(est, X_tr_d, y_tr_d, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    except Exception:
        proba_cv = None

    if proba_cv is not None:
        thr, _, prec_at_thr = tune_threshold_for_recall(y_tr_d, proba_cv)
        y_pred_cv_thr = (proba_cv >= thr).astype(int)
        mcv = metrics_binary(y_tr_d, y_pred_cv_thr, proba_cv)
    else:
        thr, prec_at_thr = np.nan, np.nan
        mcv = metrics_binary(y_tr_d, y_pred_cv, None)

    mcv.update({
        "model": name,
        "dataset": "train_cv",
        "threshold": thr,
        "precision_at_threshold": prec_at_thr,
        "best_params": str(gs.best_params_)
    })
    rows_doenca_cv.append(mcv)

    # seleção do melhor pela sensibilidade CV
    if mcv["recall"] > best_doenca_score:
        best_doenca_score = mcv["recall"]
        best_doenca_name = name
        best_doenca_estimator = est
        best_thr = thr if pd.notna(thr) else 0.5

# avaliação hold-out/teste para TODOS os modelos
for row in rows_doenca_cv:
    name = row["model"]
    est = trained_binary_estimators[name]
    est.fit(X_tr_d, y_tr_d)
    proba_te = est.predict_proba(X_te_d)[:, 1]
    thr = row["threshold"] if pd.notna(row["threshold"]) else 0.5
    y_pred_te = (proba_te >= thr).astype(int)
    mte = metrics_binary(y_te_d, y_pred_te, proba_te)
    mte.update({
        "model": name,
        "dataset": "validation_test",
        "threshold": thr,
        "precision_at_threshold": np.nan
    })
    rows_doenca_test.append(mte)

df_doenca_cv = pd.DataFrame(rows_doenca_cv).sort_values("recall", ascending=False)
df_doenca_test = pd.DataFrame(rows_doenca_test).sort_values("recall", ascending=False)

# refit best model
best_doenca_estimator.fit(X_tr_d, y_tr_d)
best_proba_te = best_doenca_estimator.predict_proba(X_te_d)[:, 1]
best_y_pred_te = (best_proba_te >= best_thr).astype(int)

# plots doença
preproc_doenca = Pipeline(best_doenca_estimator.steps[:-1])
X_tr_d_proc = preproc_doenca.fit_transform(X_tr_d, y_tr_d)
plot_pca(X_tr_d_proc, y_tr_d, "DOENCA__PCA_train.png", "PCA (train) | Disease")
save_confusion(y_te_d, best_y_pred_te, labels=[0, 1], out_png="DOENCA__confusion_matrix_test.png",
               title=f"Confusion matrix (test) | Disease | Best={best_doenca_name}",
               class_names=["Healthy", "CKD"])
plot_roc_binary(y_te_d, best_proba_te, "DOENCA__ROC_test.png",
                f"ROC (test) | Disease | Best={best_doenca_name}")

# ranking doença
feat_names_proc = transformed_feature_names(feat_cols, deriv=0)
imp_doenca = permutation_importance_df(best_doenca_estimator, X_te_d, y_te_d, feat_names_proc, scoring="recall", n_repeats=200)
sig_doenca = feature_significance_binary(X_tr_d_proc, y_tr_d, feat_names_proc)
rank_doenca = imp_doenca.merge(sig_doenca, on="feature", how="left")
plot_feature_ranking(rank_doenca, "DOENCA__feature_ranking.png", "Feature ranking | Disease")

# joblib doença (apenas melhor)
bundle_doenca = {
    "task": "doenca_binaria",
    "pipeline": best_doenca_estimator,
    "feature_cols": feat_cols,
    "threshold": float(best_thr),
    "best_model_name": best_doenca_name
}
joblib.dump(bundle_doenca, "DOENCA__best_model.joblib")

# =========================
# TASK 2: ESTÁDIOS
# =========================
y_stage_raw = df["Estadio"].astype(str).to_numpy()
le_stage = LabelEncoder()
y_stage = le_stage.fit_transform(y_stage_raw)
class_names = list(le_stage.classes_)

X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    X, y_stage, test_size=0.2, stratify=y_stage, random_state=RANDOM_STATE
)

rows_stage_cv = []
rows_stage_test = []
trained_stage_estimators = {}

best_stage_name, best_stage_score = None, -np.inf
best_stage_estimator = None

for name, (clf, grid) in model_grids_multiclass().items():
    pipe = Pipeline([
        ("pre", SavGolSNV(window=5, poly=2, deriv=0)),
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    gs = GridSearchCV(pipe, grid, scoring="recall_macro", cv=cv, n_jobs=-1, refit=True)
    gs.fit(X_tr_s, y_tr_s)
    est = gs.best_estimator_
    trained_stage_estimators[name] = est

    y_pred_cv = cross_val_predict(est, X_tr_s, y_tr_s, cv=cv, method="predict", n_jobs=-1)
    try:
        proba_cv = cross_val_predict(est, X_tr_s, y_tr_s, cv=cv, method="predict_proba", n_jobs=-1)
    except Exception:
        proba_cv = None

    mcv = metrics_multiclass(y_tr_s, y_pred_cv, proba_cv)
    mcv.update({
        "model": name,
        "dataset": "train_cv",
        "best_params": str(gs.best_params_)
    })
    rows_stage_cv.append(mcv)

    if mcv["recall_macro"] > best_stage_score:
        best_stage_score = mcv["recall_macro"]
        best_stage_name = name
        best_stage_estimator = est

# teste/hold-out para TODOS os modelos
for row in rows_stage_cv:
    name = row["model"]
    est = trained_stage_estimators[name]
    est.fit(X_tr_s, y_tr_s)
    y_pred_te_s = est.predict(X_te_s)
    try:
        proba_te_s = est.predict_proba(X_te_s)
    except Exception:
        proba_te_s = None

    mte = metrics_multiclass(y_te_s, y_pred_te_s, proba_te_s)
    mte.update({"model": name, "dataset": "validation_test"})
    rows_stage_test.append(mte)

df_stage_cv = pd.DataFrame(rows_stage_cv).sort_values("recall_macro", ascending=False)
df_stage_test = pd.DataFrame(rows_stage_test).sort_values("recall_macro", ascending=False)

# refit best stage model
best_stage_estimator.fit(X_tr_s, y_tr_s)
best_y_pred_te_s = best_stage_estimator.predict(X_te_s)
try:
    best_proba_te_s = best_stage_estimator.predict_proba(X_te_s)
except Exception:
    best_proba_te_s = None

# plots estádios
preproc_stage = Pipeline(best_stage_estimator.steps[:-1])
X_tr_s_proc = preproc_stage.fit_transform(X_tr_s, y_tr_s)
plot_pca(X_tr_s_proc, y_tr_s, "ESTADIOS__PCA_train.png", "PCA (train) | CKD stages", class_names=class_names)
save_confusion(y_te_s, best_y_pred_te_s, labels=np.arange(len(class_names)),
               out_png="ESTADIOS__confusion_matrix_test.png",
               title=f"Confusion matrix (test) | CKD stages | Best={best_stage_name}",
               class_names=class_names)

if best_proba_te_s is not None:
    plot_roc_multiclass_ovr(
        y_te_s, best_proba_te_s, class_names,
        "ESTADIOS__ROC_test.png",
        f"ROC OvR (test) | CKD stages | Best={best_stage_name}"
    )

# ranking estádios
imp_stage = permutation_importance_df(
    best_stage_estimator, X_te_s, y_te_s, feat_names_proc, scoring="recall_macro", n_repeats=200
)
sig_stage = feature_significance_multiclass(X_tr_s_proc, y_tr_s, feat_names_proc)
rank_stage = imp_stage.merge(sig_stage, on="feature", how="left")
plot_feature_ranking(rank_stage, "ESTADIOS__feature_ranking.png", "Feature ranking | CKD stages")

# joblib estádios (apenas melhor)
bundle_stage = {
    "task": "estadio_multiclasse",
    "pipeline": best_stage_estimator,
    "label_encoder": le_stage,
    "class_names": class_names,
    "feature_cols": feat_cols,
    "best_model_name": best_stage_name
}
joblib.dump(bundle_stage, "ESTADIOS__best_model.joblib")

# =========================
# EXPORT EXCEL
# =========================
OUT_XLSX = "MININIR__classification_metrics_revised.xlsx"

with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    # folhas completas
    df_doenca_cv.to_excel(xw, index=False, sheet_name="DOENCA_trainCV_all")
    df_doenca_test.to_excel(xw, index=False, sheet_name="DOENCA_validation_all")
    rank_doenca.to_excel(xw, index=False, sheet_name="DOENCA_features")

    df_stage_cv.to_excel(xw, index=False, sheet_name="ESTADIOS_trainCV_all")
    df_stage_test.to_excel(xw, index=False, sheet_name="ESTADIOS_validation_all")
    rank_stage.to_excel(xw, index=False, sheet_name="ESTADIOS_features")

    # resumo manuscrito
    resumo_doenca = df_doenca_cv[["model", "roc_auc", "accuracy", "precision", "recall", "f1_score", "specificity"]].merge(
        df_doenca_test[["model", "roc_auc", "accuracy", "precision", "recall", "f1_score", "specificity"]],
        on="model", suffixes=("_trainCV", "_validation")
    )
    resumo_stage = df_stage_cv[["model", "roc_auc_ovr_weighted", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]].merge(
        df_stage_test[["model", "roc_auc_ovr_weighted", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]],
        on="model", suffixes=("_trainCV", "_validation")
    )
    resumo_doenca.to_excel(xw, index=False, sheet_name="SUMMARY_DOENCA")
    resumo_stage.to_excel(xw, index=False, sheet_name="SUMMARY_ESTADIOS")

print("OK ✅")
print("Saved:", OUT_XLSX)
print("Saved best joblib: DOENCA__best_model.joblib, ESTADIOS__best_model.joblib")
print("Saved plots:")
print(" - DOENCA__PCA_train.png")
print(" - DOENCA__confusion_matrix_test.png")
print(" - DOENCA__ROC_test.png")
print(" - DOENCA__feature_ranking.png")
print(" - ESTADIOS__PCA_train.png")
print(" - ESTADIOS__confusion_matrix_test.png")
if best_proba_te_s is not None:
    print(" - ESTADIOS__ROC_test.png")
print(" - ESTADIOS__feature_ranking.png")
print("Best (Doença):", best_doenca_name, "| threshold:", best_thr)
print("Best (Estádios):", best_stage_name)

if IN_COLAB:
    for fn in [
        OUT_XLSX,
        "DOENCA__PCA_train.png", "DOENCA__confusion_matrix_test.png", "DOENCA__ROC_test.png", "DOENCA__feature_ranking.png",
        "ESTADIOS__PCA_train.png", "ESTADIOS__confusion_matrix_test.png", "ESTADIOS__feature_ranking.png",
        "DOENCA__best_model.joblib", "ESTADIOS__best_model.joblib"
    ]:
        if os.path.exists(fn):
            files.download(fn)
    if os.path.exists("ESTADIOS__ROC_test.png"):
        files.download("ESTADIOS__ROC_test.png")
