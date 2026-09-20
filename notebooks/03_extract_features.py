"""
Step 3: Extract features (S3).

Reads each split's per-sample dumps (samples/{split}/{id}.npz) and writes two feature
frames per split (no labels here - labels come from verification/ and are joined in step 5):

  - features/{split}_attention.parquet   : wide dens_*/ll_lookback_* attention features + id
  - features/{split}_probability.parquet : probability-score features + id

Attention features: mean over response tokens of the per-token category densities, the way
Lookback Lens normalizes them - dens_* (the five categories) and ll_lookback_* (their two-group
prompt-vs-output ratio), per layer/head. Step 5 drops dens_output as the reference category.
Probability features summarise the per-token generation probabilities p_t.

Both are derived from the dumped cat_ratios rather than the raw attention maps, so this step
onward is enough to rebuild them - no GPU rerun of step 02.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path("..").resolve()))
from src.attention_features_v2 import features_from_densities, region_sizes
import aws_config as cfg


def _score_features(scores):
    """Scalar summaries of a sample's per-token probabilities p_t."""
    logp = np.log(scores)
    return {
        "normalized_score": logp.sum() / len(scores),   # mean log-prob per token
        "prod_score": float(np.prod(scores)),           # sequence likelihood
        "min_score": float(np.min(scores)),             # least-confident token
        "std_logp": float(np.std(logp)),                # spread / unevenness of confidence
        "len_output": len(scores),                      # number of generated tokens
    }


def build_features(split):
    ids = pd.read_parquet(cfg.generation_uri(split))["id"].unique()
    attn_rows, prob_rows = [], []
    for sid in ids:
        npz = cfg.load_npz(cfg.sample_uri(split, sid))
        ratios, sizes = npz["cat_ratios"], region_sizes(npz["input_tokens"])
        attn_rows.append({**features_from_densities(ratios, sizes),     # dens_*, ll_lookback_*
                          "id": sid})
        prob_rows.append({**_score_features(npz["scores"]), "id": sid})

    pd.DataFrame(attn_rows).to_parquet(cfg.attention_features_uri(split), index=False)
    pd.DataFrame(prob_rows).to_parquet(cfg.probability_features_uri(split), index=False)
    print(f"[{split}] attention {len(attn_rows)} rows, probability {len(prob_rows)} rows")


for split in cfg.SPLITS:
    build_features(split)
