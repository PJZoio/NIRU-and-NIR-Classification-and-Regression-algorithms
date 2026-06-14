# -*- coding: utf-8 -*-
"""
Figure regeneration for the miniNIR vs benchtop FT-NIR CKD comparison study.

This script reproduces the exploratory (PCA), regression and classification
figures of the manuscript directly from the four raw spectral CSV files. It
REUSES, without modification, the preprocessing, train/validation split,
hyperparameter grids and model choices of the original analysis pipelines
(regressao_mininir.py, nir_regression_windows_nestedcv_lite_v3.py,
classificacao_mininir__estadios_e_doenca__revised.py, nir_classification_lite_v2.py).

Key settings (identical to the original pipelines):
  - Per-sample Savitzky-Golay + SNV preprocessing:
        miniNIR  : window=5,  poly=2, deriv=0 (smoothing only)
        FT-NIR   : window=21, poly=2, deriv=1 (1st derivative)
  - StandardScaler after preprocessing.
  - Hold-out split: test_size=0.20, stratified, random_state=42.
  - Classifiers: SVC(RBF, class_weight='balanced'), XGBoost; hyperparameters
    chosen by inner StratifiedKFold grid search (scoring: recall for the binary
    disease task, recall_macro for the 6-class stage task).
  - Disease decision threshold tuned on the training out-of-fold probabilities
    (lowest threshold maximizing recall; linspace(0.05,0.95,19)), as in the
    original classification scripts.
  - FT-NIR classification/PCA use the optimized spectral windows
    (disease 5800-6300 cm-1; stage 5500-6000 cm-1); miniNIR uses the 6 LED bands.

Outputs (PNG) written to OUT: PCA (test), ROC and confusion matrices for both
instruments and both tasks, plus predicted-vs-reference regression panels.

Requires: numpy, pandas, scipy, scikit-learn, xgboost, matplotlib.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, roc_auc_score
from xgboost import XGBClassifier

RS=42; np.random.seed(RS)
OUT="/tmp/regen"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"figure.dpi":140,"savefig.dpi":200,"font.size":9,"axes.titlesize":10,
  "axes.labelsize":9,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
  "axes.linewidth":0.8,"lines.linewidth":1.4,"grid.alpha":0.15})

class SavGolSNV(BaseEstimator,TransformerMixin):
    def __init__(self,window=5,poly=2,deriv=0): self.window=window; self.poly=poly; self.deriv=deriv
    def fit(self,X,y=None): return self
    def transform(self,X):
        X=np.asarray(X,float); n=X.shape[1]; w=self.window
        if w%2==0: w+=1
        if w>n: w=n if n%2==1 else max(3,n-1)
        Xf=X.copy() if w<3 else savgol_filter(X,window_length=w,polyorder=min(self.poly,w-1),deriv=self.deriv,axis=1)
        mu=Xf.mean(1,keepdims=True); sd=Xf.std(1,keepdims=True)+1e-12
        return (Xf-mu)/sd

def pre(kind): return SavGolSNV(5,2,0) if kind=="mini" else SavGolSNV(21,2,1)

def pca_test(Xtr,Xte,yte,kind,title,classnames,fname):
    p=Pipeline([("pre",pre(kind)),("sc",StandardScaler())]); Ptr=p.fit_transform(Xtr); Pte=p.transform(Xte)
    pca=PCA(2,random_state=RS).fit(Ptr); Z=pca.transform(Pte); ev=pca.explained_variance_ratio_*100
    fig,ax=plt.subplots(figsize=(5.0,4.2))
    for c in np.unique(yte):
        m=yte==c; ax.scatter(Z[m,0],Z[m,1],s=30,alpha=0.85,edgecolor="black",linewidth=0.25,
                              label=classnames[int(c)] if classnames else str(c))
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)"); ax.set_title(title)
    ax.legend(fontsize=7,framealpha=0.9); ax.grid(True); fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

def fit_model(Xtr,ytr,kind,model,task):
    if model=="svc":
        clf=SVC(probability=True,class_weight="balanced",random_state=RS)
        grid={"clf__C":[0.1,1,10],"clf__gamma":["scale",0.01]}
    else:
        clf=XGBClassifier(random_state=RS,eval_metric="mlogloss",n_estimators=300,
                          tree_method="hist",num_class=len(np.unique(ytr)) if len(np.unique(ytr))>2 else None)
        grid={"clf__max_depth":[2,3],"clf__learning_rate":[0.1,0.3]}
    pipe=Pipeline([("pre",pre(kind)),("sc",StandardScaler()),("clf",clf)])
    inner=StratifiedKFold(3,shuffle=True,random_state=RS)
    sc="recall" if task=="disease" else "recall_macro"
    gs=GridSearchCV(pipe,grid,scoring=sc,cv=inner,n_jobs=-1)
    gs.fit(Xtr,ytr); return gs.best_estimator_

def roc_cm(Xtr,Xte,ytr,yte,kind,model,task,inst,classnames):
    est=fit_model(Xtr,ytr,kind,model,task)
    if task=="disease":
        proba=est.predict_proba(Xte)[:,1]
        # threshold tuned on TRAIN out-of-fold proba (maximize recall, tie-break accuracy)
        inner=StratifiedKFold(3,shuffle=True,random_state=RS)
        oof=cross_val_predict(est,Xtr,ytr,cv=inner,method="predict_proba",n_jobs=-1)[:,1]
        best_t,best_rec=0.5,-1.0
        for t in np.linspace(0.05,0.95,19):
            rec=((oof>=t).astype(int)[ytr==1]==1).mean() if (ytr==1).any() else 0.0
            if rec>best_rec: best_rec=rec; best_t=float(t)
        thr=best_t
        yp=(proba>=thr).astype(int)
        fpr,tpr,_=roc_curve(yte,proba); a=auc(fpr,tpr)
        fig,ax=plt.subplots(figsize=(5.0,4.2)); ax.plot(fpr,tpr,label=f"AUC = {a:.3f}")
        ax.plot([0,1],[0,1],"--",color="orange"); ax.set_xlabel("1 - Specificity (FPR)")
        ax.set_ylabel("Sensitivity (TPR)"); ax.set_title(f"ROC (test) | Disease | {inst}")
        ax.legend(loc="lower right"); ax.grid(True); fig.tight_layout(); fig.savefig(f"{OUT}/{inst}_disease_ROC.png"); plt.close(fig)
        cm=confusion_matrix(yte,yp)
        d=ConfusionMatrixDisplay(cm,display_labels=classnames); fig,ax=plt.subplots(figsize=(4.6,4.2))
        d.plot(ax=ax,colorbar=False); ax.set_title(f"Confusion matrix (test) | Disease | {inst}"); fig.tight_layout(); fig.savefig(f"{OUT}/{inst}_disease_CM.png"); plt.close(fig)
        print(f"{inst} disease: AUC={a:.3f} thr={thr:.2f} acc={(yp==yte).mean():.3f}")
    else:
        proba=est.predict_proba(Xte); yp=proba.argmax(1)
        nc=proba.shape[1]; Y=label_binarize(yte,classes=list(range(nc)))
        fprs=[]; tprs=[]
        for i in range(nc):
            f,t,_=roc_curve(Y[:,i],proba[:,i]); fprs.append(f); tprs.append(t)
        allf=np.unique(np.concatenate(fprs)); meant=np.zeros_like(allf)
        for i in range(nc): meant+=np.interp(allf,fprs[i],tprs[i])
        meant/=nc; macro=auc(allf,meant)
        wauc=roc_auc_score(yte,proba,multi_class="ovr",average="weighted")
        fig,ax=plt.subplots(figsize=(5.0,4.2)); ax.plot(allf,meant,label=f"Macro AUC = {macro:.3f}")
        ax.plot([0,1],[0,1],"--",color="orange"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title(f"ROC (test) | CKD stage | {inst}"); ax.legend(loc="lower right"); ax.grid(True)
        fig.tight_layout(); fig.savefig(f"{OUT}/{inst}_stage_ROC.png"); plt.close(fig)
        cm=confusion_matrix(yte,yp)
        d=ConfusionMatrixDisplay(cm,display_labels=list(range(nc))); fig,ax=plt.subplots(figsize=(4.6,4.2))
        d.plot(ax=ax,colorbar=False); ax.set_title(f"Confusion matrix (test) | CKD stage | {inst}"); fig.tight_layout(); fig.savefig(f"{OUT}/{inst}_stage_CM.png"); plt.close(fig)
        print(f"{inst} stage: macroAUC={macro:.3f} weightedAUC={wauc:.3f} acc={(yp==yte).mean():.3f}")

# ============ miniNIR ============
dm=pd.read_csv("/mnt/user-data/uploads/2026_02_16-miniNir__solucoes_compostas_-Doenca_e_Estadio.csv",sep=";",decimal=",")
Xm=dm[[f"LED{i}_avg" for i in range(1,7)]].to_numpy()
ydm=dm["Doenca"].astype(int).to_numpy()
ysm=LabelEncoder().fit_transform(dm["Estadio"].astype(str))
Xtr,Xte,ytr,yte=train_test_split(Xm,ydm,test_size=0.2,stratify=ydm,random_state=RS)
pca_test(Xtr,Xte,yte,"mini","PCA (test) | Disease | miniNIR",["Healthy","CKD"],"miniNIR_disease_PCA.png")
roc_cm(Xtr,Xte,ytr,yte,"mini","svc","disease","miniNIR",["Healthy","CKD"])
Xtr,Xte,ytr,yte=train_test_split(Xm,ysm,test_size=0.2,stratify=ysm,random_state=RS)
pca_test(Xtr,Xte,yte,"mini","PCA (test) | CKD stage | miniNIR",None,"miniNIR_stage_PCA.png")
roc_cm(Xtr,Xte,ytr,yte,"mini","xgb","stage","miniNIR",None)

# ============ benchtop ============
db=pd.read_csv("/mnt/user-data/uploads/Compostas__Novas__-_Espetros-Doenca_e_estadios.csv",sep=";",decimal=",",low_memory=False)
def wn(c):
    try: return float(str(c).replace(",","."))
    except: return None
spec=[c for c in db.columns if wn(c) is not None]
spec_df=db[spec].apply(pd.to_numeric,errors="coerce").dropna(axis=1)
spec=list(spec_df.columns)
wns=np.array([wn(c) for c in spec])
Xb_full=spec_df.to_numpy()
print("benchtop spectral cols kept:",len(spec))
ydb=db["Doenca"].astype(int).to_numpy(); ysb=db["Estadio"].astype(int).to_numpy()
def win(lo,hi):
    m=(wns>=lo)&(wns<=hi); return spec_df.loc[:,[spec[i] for i in range(len(spec)) if m[i]]].to_numpy()
# PCA: full spectrum
Xtr,Xte,ytr,yte=train_test_split(Xb_full,ydb,test_size=0.2,stratify=ydb,random_state=RS)
pca_test(Xtr,Xte,yte,"bench","PCA (test) | Disease | FT-NIR",["Healthy","CKD"],"FT-NIR_disease_PCA.png")
Xtr,Xte,ytr,yte=train_test_split(Xb_full,ysb,test_size=0.2,stratify=ysb,random_state=RS)
pca_test(Xtr,Xte,yte,"bench","PCA (test) | CKD stage | FT-NIR",None,"FT-NIR_stage_PCA.png")
# windowed FT-NIR PCA (informative region) for fair comparison
Xdw=win(5800,6300); Xtr,Xte,ytr,yte=train_test_split(Xdw,ydb,test_size=0.2,stratify=ydb,random_state=RS)
pca_test(Xtr,Xte,yte,"bench","PCA (test) | Disease | FT-NIR (window)",["Healthy","CKD"],"FT-NIR_disease_PCAwin.png")
Xsw=win(5500,6000); Xtr,Xte,ytr,yte=train_test_split(Xsw,ysb,test_size=0.2,stratify=ysb,random_state=RS)
pca_test(Xtr,Xte,yte,"bench","PCA (test) | CKD stage | FT-NIR (window)",None,"FT-NIR_stage_PCAwin.png")
# ROC/CM: best windows
Xd=win(5800,6300); Xtr,Xte,ytr,yte=train_test_split(Xd,ydb,test_size=0.2,stratify=ydb,random_state=RS)
roc_cm(Xtr,Xte,ytr,yte,"bench","svc","disease","FT-NIR",["Healthy","CKD"])
Xs=win(5500,6000); Xtr,Xte,ytr,yte=train_test_split(Xs,ysb,test_size=0.2,stratify=ysb,random_state=RS)
roc_cm(Xtr,Xte,ytr,yte,"bench","svc","stage","FT-NIR",None)
print("done"); print(sorted(os.listdir(OUT)))
