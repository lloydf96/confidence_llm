"""
Step 4: Verify answers (exact match first, Bedrock on a miss).

Reads generation/{split}.parquet and writes verification/{split}.parquet with:
  id, is_correct, verify_method   (verify_method in {"exact", "api"})

Exact match settles most answers for free; only the rest hit Amazon Bedrock (Nova Lite).
Resumable: ids already in the verification parquet are skipped.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path("..").resolve()))
from src.verification_bedrock import BedrockVerifier, verify
import aws_config as cfg

verifier = BedrockVerifier(model_id=cfg.BEDROCK_MODEL_ID, region=cfg.BEDROCK_REGION)


def _read_parquet_or_none(uri):
    try:
        return pd.read_parquet(uri)
    except (FileNotFoundError, OSError):
        return None


def verify_split(split):
    gen = pd.read_parquet(cfg.generation_uri(split))

    done = _read_parquet_or_none(cfg.verification_uri(split))
    rows = done.to_dict("records") if done is not None else []
    done_ids = set(done["id"]) if done is not None else set()
    todo = gen[~gen["id"].isin(done_ids)].reset_index(drop=True)
    print(f"[{split}] {len(todo)} to verify, {len(done_ids)} already done")

    for i, r in enumerate(todo.itertuples()):
        is_correct, method = verify(
            verifier, r.context, r.question, r.answer_text, r.generated_answer)
        rows.append({"id": r.id, "is_correct": bool(is_correct), "verify_method": method})

        if (i + 1) % 100 == 0:
            pd.DataFrame(rows).to_parquet(cfg.verification_uri(split), index=False)
            print(f"[{split}] verified {i + 1}/{len(todo)}")

    out = pd.DataFrame(rows)
    out.to_parquet(cfg.verification_uri(split), index=False)
    api = int((out["verify_method"] == "api").sum())
    print(f"[{split}] done: {len(out)} labels, {api} via API, {len(out) - api} via exact match")


for split in cfg.SPLITS:
    verify_split(split)
