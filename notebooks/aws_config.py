"""Config + S3 helpers for the AWS confidence-llm pipeline.

Optional env overrides (via confidence_llm/.env): CONFIDENCE_LLM_BUCKET, CONFIDENCE_LLM_PREFIX,
AWS_REGION, BEDROCK_MODEL_ID, GEN_BATCH_SIZE, MAX_NEW_TOKENS, HF_TOKEN.
S3 layout: raw/ samples/ generation/ verification/ features/ models/ results/.
"""
import io
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:


    
    pass

import boto3
import numpy as np

REGION = os.getenv("AWS_REGION") or boto3.session.Session().region_name or "us-east-1"
PREFIX = os.getenv("CONFIDENCE_LLM_PREFIX", "confidence-llm").strip("/")
BUCKET = os.getenv("CONFIDENCE_LLM_BUCKET") or \
    f"sagemaker-{REGION}-{boto3.client('sts').get_caller_identity()['Account']}"

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", REGION)
MODEL_NAME = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))
BATCH_SIZE = int(os.getenv("GEN_BATCH_SIZE", "8"))
SPLITS = ("train", "test")

# Models: name -> (feature blocks, label).
# Attention blocks are length-normalized the Lookback Lens way - attention per token, so the
# feature does not move with prompt or answer length: ll_lookback (Chuang et al.'s two-group
# ratio per layer/head); dens_split (the four input-category densities per layer/head).
# See src/attention_features_v2.
MODEL_CONFIGS = {
    "prob":             (["prob"],                "Pure probability"),
    "ll_lookback":      (["ll_lookback"],         "Lookback Lens ratio only"),
    "prob_ll_lookback": (["prob", "ll_lookback"], "Probability + Lookback Lens ratio"),
    "dens_split":       (["dens_split"],          "Attention densities only (split)"),
    "prob_dens_split":  (["prob", "dens_split"],  "Probability + attention densities (split)"),
}

# --- S3 paths ---
def s3_uri(*parts):
    return f"s3://{BUCKET}/" + "/".join([PREFIX, *(str(p).strip("/") for p in parts)])

def raw_uri(split):                  return s3_uri("raw", f"{split}.parquet")
def generation_uri(split):           return s3_uri("generation", f"{split}.parquet")
def verification_uri(split):         return s3_uri("verification", f"{split}.parquet")
def attention_features_uri(split):   return s3_uri("features", f"{split}_attention.parquet")
def probability_features_uri(split): return s3_uri("features", f"{split}_probability.parquet")
def sample_uri(split, sid):          return s3_uri("samples", split, f"{sid}.npz")
def model_uri(name):                 return s3_uri("models", name)
def results_uri(name):               return s3_uri("results", name)

# --- S3 IO (parquet goes through pandas/s3fs directly) ---
_s3 = boto3.client("s3", region_name=REGION)

def _key(uri):
    b, _, k = uri.partition("s3://")[2].partition("/")
    return b, k

def _put(uri, data):
    b, k = _key(uri); _s3.put_object(Bucket=b, Key=k, Body=data)

def _get(uri):
    b, k = _key(uri); return _s3.get_object(Bucket=b, Key=k)["Body"].read()

def s3_exists(uri):
    b, k = _key(uri)
    try:
        _s3.head_object(Bucket=b, Key=k); return True
    except _s3.exceptions.ClientError:
        return False

def save_npz(uri, **arrays):
    buf = io.BytesIO(); np.savez(buf, **arrays); _put(uri, buf.getvalue())

def load_npz(uri):
    return np.load(io.BytesIO(_get(uri)), allow_pickle=True)

def save_joblib(uri, obj):
    import joblib
    buf = io.BytesIO(); joblib.dump(obj, buf); _put(uri, buf.getvalue())

def load_joblib(uri):
    import joblib
    return joblib.load(io.BytesIO(_get(uri)))
