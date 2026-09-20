"""
Step 5: Train the confidence models (S3).

Trains the models in `cfg.MODEL_CONFIGS` from three feature blocks:
  - prob        : probability-score features
  - ll_lookback : per-(layer, head) Lookback Lens ratio — the whole prompt against the model's
                  own output, length-normalized (attention per token, not per group)
  - dens_split  : the four input-category densities per (layer, head)

Each model uses the same procedure: empty-start forward selection (cap 25) on CV ROC-AUC over
its candidate columns, then an L1 logistic fit. Writes, per model `<name>`:
  - models/<name>.joblib
  - results/{split}_<name>.parquet : id, pred, is_correct
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed

sys.path.append(str(Path("..").resolve()))
import aws_config as cfg

warnings.filterwarnings("ignore")

PROB_FEATS = ["normalized_score", "prod_score", "min_score", "std_logp", "len_output"]
INPUT_CATS = ["context", "question", "special", "instruct"]
C_VALUE = 1e-2
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def load_split(split):
    """Features (probability + the length-normalized attention family) joined with labels."""
    attn = pd.read_parquet(cfg.attention_features_uri(split)).drop_duplicates("id").set_index("id")
    # keep the Lookback Lens family only, and drop dens_output as the reference category
    # (the five densities sum to 1, so keeping all five would be a dummy-variable trap)
    attn = attn[[c for c in attn.columns
                 if (c.startswith("dens_") and not c.startswith("dens_output_"))
                 or c.startswith("ll_lookback_head_")]]
    prob = pd.read_parquet(cfg.probability_features_uri(split)).drop_duplicates("id").set_index("id")[PROB_FEATS]
    labels = (pd.read_parquet(cfg.verification_uri(split)).drop_duplicates("id")
              .set_index("id")["is_correct"].astype(int))
    X = attn.join(prob, how="inner")
    X, y = X.align(labels, join="inner", axis=0)
    return X.fillna(0).sort_index(), y.sort_index()


X_train, y_train = load_split("train")
X_test, y_test = load_split("test")
X_test = X_test[X_train.columns]
y_arr = y_train.values

BLOCKS = {
    "prob": PROB_FEATS,
    "ll_lookback": [c for c in X_train.columns if c.startswith("ll_lookback_head_")],
    "dens_split": [c for c in X_train.columns
                   if any(c.startswith(f"dens_{k}_head_") for k in INPUT_CATS)],
}
print(f"Train X={X_train.shape}  blocks: "
      + ", ".join(f"{k}={len(v)}" for k, v in BLOCKS.items()))


def make_clf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear", C=C_VALUE,
                                   max_iter=1000, class_weight="balanced")),
    ])


def cv_auc(arr):
    return cross_val_score(make_clf(), arr, y_arr, scoring="roc_auc", cv=CV, n_jobs=1).mean()


def forward_select(cands, cap=25):
    """Empty-start greedy selection over candidate columns; returns chosen names + CV AUC."""
    arr = np.ascontiguousarray(X_train[cands].values)
    chosen_pos, remaining, best, chosen = [], list(range(len(cands))), 0.0, []
    for _ in range(min(cap, len(cands))):
        scores = Parallel(n_jobs=-1)(delayed(cv_auc)(arr[:, chosen_pos + [j]]) for j in remaining)
        p = int(np.argmax(scores))
        if scores[p] <= best:
            break
        j = remaining[p]
        chosen_pos.append(j); remaining.remove(j); best = scores[p]; chosen.append(cands[j])
    return chosen, best


def fit_final(cols):
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegressionCV(penalty="l1", solver="saga", Cs=[C_VALUE], cv=CV,
                                     scoring="roc_auc", max_iter=5000, n_jobs=-1,
                                     refit=True, class_weight="balanced")),
    ])
    m.fit(X_train[cols], y_train)
    return m


for name, (blocks, label) in cfg.MODEL_CONFIGS.items():
    cands = [c for b in blocks for c in BLOCKS[b]]
    chosen, cv = forward_select(cands)
    model = fit_final(chosen)
    cfg.save_joblib(cfg.model_uri(f"{name}.joblib"), model)
    for split, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
        pred = model.predict_proba(X[chosen])[:, 1]
        (pd.DataFrame({"id": X.index, "pred": pred, "is_correct": y.values})
         .to_parquet(cfg.results_uri(f"{split}_{name}.parquet"), index=False))
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test[chosen])[:, 1])
    print(f"  {name:14s} {len(chosen):2d} feats  CV={cv:.4f}  test={test_auc:.4f}  ({label})")
