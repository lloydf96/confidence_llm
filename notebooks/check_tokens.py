"""
Check that region_masks classifies prompt tokens into the right categories - WITHOUT running
generation (CPU only, no GPU, no S3 dumps needed).

Tokenizes a prompt exactly like 02_run_llm.py does (chat template), applies region_masks, and
prints an indexed  token -> category  dump so you can eyeball that context/question/special/
instruct land on the right tokens. Run this before 02 to confirm the masks are aligned in your
environment.

Run:  python check_tokens.py            # uses the first row of raw/test.parquet
      python check_tokens.py train
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

sys.path.append(str(Path("..").resolve()))
from src.attention_features_v2 import region_masks, assert_prompt_layout
import aws_config as cfg

# Fallback if raw/{split}.parquet isn't reachable.
EXAMPLE = {
    "context": ("The Amazon rainforest covers most of the Amazon basin of South America. This "
                "region includes territory belonging to nine nations and 3,344,000 square "
                "kilometres are in Brazil."),
    "question": "How many nations does the Amazon basin territory belong to?",
}


def chat_prompt(tok, context, question):
    text = ("Look at the context given below and answer the question."
            f"Answer in as few words as possible.\nContext: {context}\nQuestion: {question}")
    return tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def main(split="test"):
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    try:
        row = pd.read_parquet(cfg.raw_uri(split)).iloc[0]
        ex = {"context": row["context"], "question": row["question"]}
        print(f"using first row of {cfg.raw_uri(split)}")
    except Exception as e:
        ex = EXAMPLE
        print(f"(couldn't read raw parquet: {e}; using the built-in example)")

    prompt = chat_prompt(tok, ex["context"], ex["question"])
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    toks = tok.convert_ids_to_tokens(ids)

    m = region_masks(toks)
    lab = np.array(["-"] * len(toks), dtype=object)
    for name in ("context", "question", "special", "instruct"):
        lab[m[name]] = name

    print(f"\n{len(toks)} prompt tokens\n")
    print(f"{'idx':>4}  {'region':8}  token")
    for i, (t, l) in enumerate(zip(toks, lab)):
        print(f"{i:>4}  {l:8}  {t}")

    print("\ncounts:", {k: int(m[k].sum()) for k in ("context", "question", "special", "instruct")})
    print("context (first 10):", [toks[i] for i in np.where(m["context"])[0][:10]])
    print("question:          ", [toks[i] for i in np.where(m["question"])[0]])
    print("special:           ", [toks[i] for i in np.where(m["special"])[0]])

    try:
        assert_prompt_layout(toks)
        print("\nassert_prompt_layout: PASS - masks are aligned, safe to run 02")
    except AssertionError as e:
        print(f"\nassert_prompt_layout: FAIL - {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
