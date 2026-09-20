"""
Step 2: Run LLM and extract attention (batched, S3).

For each sample, generates an answer with Qwen2.5-0.5B-Instruct and writes to S3:
  - samples/{split}/{id}.npz : cat_ratios (n_gen, 24, 14, 5), scores, input/output tokens
  - generation/{split}.parquet : id, generated_answer, context, question, answer_text

Generation is batched (GEN_BATCH_SIZE) for speed and resumable (samples already in the
generation parquet are skipped). Needs a GPU + HF access; `attn_implementation="eager"` and
`output_attentions=True` are required to read attention weights.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path("..").resolve()))
from src.attention_features_v2 import sample_category_ratios, assert_prompt_layout
import aws_config as cfg

_aligned = False   # region-mask alignment is checked once, on the first sample

# --- Model + tokenizer -------------------------------------------------------
# HF token from the environment (SageMaker is headless, so no notebook_login).
from huggingface_hub import login
if os.environ.get("HF_TOKEN"):
    login(token=os.environ["HF_TOKEN"])

tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
tokenizer.padding_side = "left"                     # decoder-only batched generation
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    cfg.MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="eager",                    # required to extract attention weights
)
print(f"Model loaded: {cfg.MODEL_NAME} | batch={cfg.BATCH_SIZE} | bucket={cfg.BUCKET}")


def _read_parquet_or_none(uri):
    try:
        return pd.read_parquet(uri)
    except (FileNotFoundError, OSError):
        return None


def chat_prompt(context, question):
    """Same prompt + Qwen chat template as the original run (region_masks assumes this layout)."""
    text = ("Look at the context given below and answer the question."
            f"Answer in as few words as possible.\nContext: {context}\nQuestion: {question}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)


def process_split(split):
    raw = pd.read_parquet(cfg.raw_uri(split))

    done = _read_parquet_or_none(cfg.generation_uri(split))
    rows = done.to_dict("records") if done is not None else []
    done_ids = set(done["id"]) if done is not None else set()
    todo = raw[~raw["id"].isin(done_ids)].reset_index(drop=True)
    print(f"[{split}] {len(todo)} to process, {len(done_ids)} already done")

    for start in range(0, len(todo), cfg.BATCH_SIZE):
        batch = todo.iloc[start:start + cfg.BATCH_SIZE]
        prompts = [chat_prompt(r.context, r.question) for r in batch.itertuples()]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=cfg.MAX_NEW_TOKENS,
                do_sample=False,            # greedy: reproducible + true (unwarped) token probabilities
                use_cache=True,             # decode-step attentions have query_len == 1
                output_attentions=True,
                output_scores=True,
                return_dict_in_generate=True,
            )

        padded_len = enc.input_ids.shape[1]
        for i, r in enumerate(batch.itertuples()):
            real_len = int(enc.attention_mask[i].sum())
            pad_len = padded_len - real_len
            input_tokens = tokenizer.convert_ids_to_tokens(
                enc.input_ids[i, pad_len:].tolist())

            global _aligned
            if not _aligned:                # abort early (seconds) if masks don't fit the prompt
                assert_prompt_layout(input_tokens)
                _aligned = True
                print("[align] region masks match the prompt layout")

            gen_ids = gen.sequences[i, padded_len:].tolist()
            n_gen = gen_ids.index(tokenizer.pad_token_id) if tokenizer.pad_token_id in gen_ids else len(gen_ids)
            n_gen = max(n_gen, 1)
            gen_ids = gen_ids[:n_gen]

            scores = np.array(
                [F.softmax(gen.scores[t][i], dim=-1)[gen_ids[t]].item() for t in range(n_gen)],
                dtype=np.float32,
            )
            cat_ratios = sample_category_ratios(
                gen.attentions, i, pad_len, real_len, n_gen, input_tokens).astype(np.float32)

            cfg.save_npz(
                cfg.sample_uri(split, r.id),
                cat_ratios=cat_ratios,
                scores=scores,
                input_tokens=np.array(input_tokens),
                output_tokens=np.array(tokenizer.convert_ids_to_tokens(gen_ids)),
            )
            rows.append({
                "id": r.id,
                "generated_answer": tokenizer.decode(gen_ids, skip_special_tokens=True),
                "context": r.context,
                "question": r.question,
                "answer_text": r.answer_text,
            })

        pd.DataFrame(rows).to_parquet(cfg.generation_uri(split), index=False)  # checkpoint
        print(f"[{split}] processed {min(start + cfg.BATCH_SIZE, len(todo))}/{len(todo)}")

    print(f"[{split}] done -> {cfg.generation_uri(split)}")


for split in cfg.SPLITS:
    process_split(split)
