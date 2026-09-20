"""
Tests for the v2 feature calculation (attention category ratios + probability features)
and the exact-match verification short-circuit.

Run:  python -m pytest tests/test_features.py    (or)    python tests/test_features.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.attention_features_v2 import (
    CATEGORIES, region_masks, sample_category_ratios, per_token_category_ratios,
    assert_prompt_layout, find_loc, region_sizes, density_ratios, features_from_densities,
)
from src.verification_bedrock import exact_match, verify


def _chat_tokens(n_sys=16):
    """A chat-templated prompt like Qwen's; n_sys = system-message length (varies by tokenizer
    version, which shifts the whole prefix). Context is always 7 tokens, question 5 (incl marker)."""
    toks  = ["<|im_start|>", "system", "\n"]                        # special header
    toks += [f"sys{i}" for i in range(n_sys)]                        # instruct (system msg)
    toks += ["<|im_end|>", "\n", "<|im_start|>", "user", "\n"]       # special
    toks += [f"ins{i}" for i in range(17)]                           # instruct (instruction)
    toks += ["Context", ":"] + [f"ctx{i}" for i in range(7)]         # marker + context
    toks += ["Question", ":"] + [f"q{i}" for i in range(3)]          # marker + question
    toks += ["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n"]  # special tail
    return toks


INPUT_TOKENS = _chat_tokens()          # n=60: context at 43:50, question at 50:55
N = len(INPUT_TOKENS)
COUNTS = {"context": 7, "question": 5, "special": 8, "instruct": 40}


def _uniform_attention(pad_len):
    """Two generated steps of attention, uniform over real keys, zero on left-pad keys."""
    heads = 2
    kp = pad_len + N                      # prefill key length
    prefill = np.zeros((1, heads, kp, kp))
    prefill[:, :, :, pad_len:] = 1.0 / N
    kd = pad_len + N + 1                  # decode key length (+1 generated token)
    decode = np.zeros((1, heads, 1, kd))
    decode[:, :, :, pad_len:] = 1.0 / (N + 1)
    return ((prefill,), (decode,))       # one layer per step


def test_region_masks_partition():
    m = region_masks(INPUT_TOKENS)
    for name, count in COUNTS.items():
        assert m[name].sum() == count, name
    stacked = np.stack([m[k] for k in COUNTS])
    assert (stacked.sum(axis=0) == 1).all()          # disjoint and covers every position


def test_category_ratios_values():
    ratios = sample_category_ratios(_uniform_attention(0), 0, 0, N, 2, INPUT_TOKENS)
    assert ratios.shape == (2, 1, 2, 5)
    # ratios sum to ~1 across the 5 categories for every token/layer/head
    assert np.allclose(ratios.sum(axis=-1), 1.0)
    prefill_expected = [COUNTS[c] / N for c in ("context", "question", "special", "instruct")] + [0.0]
    decode_expected = [COUNTS[c] / (N + 1) for c in ("context", "question", "special", "instruct")] + [1.0 / (N + 1)]
    assert np.allclose(ratios[0, 0, 0], prefill_expected)
    assert np.allclose(ratios[1, 0, 0], decode_expected)


def test_padding_invariance():
    """Left-padding must not change the ratios (pad keys carry ~zero attention)."""
    unpadded = sample_category_ratios(_uniform_attention(0), 0, 0, N, 2, INPUT_TOKENS)
    padded = sample_category_ratios(_uniform_attention(4), 0, 4, N, 2, INPUT_TOKENS)
    assert np.allclose(unpadded, padded)


def test_per_token_wrapper_matches():
    a = _uniform_attention(0)
    assert np.allclose(per_token_category_ratios(a, N, INPUT_TOKENS),
                       sample_category_ratios(a, 0, 0, N, len(a), INPUT_TOKENS))


def _uniform_attention_n(n_gen, tokens=INPUT_TOKENS, pad_len=0):
    """`n_gen` steps of attention, uniform over every real key (i.e. no preference at all)."""
    n, heads = len(tokens), 2
    steps = []
    for t in range(n_gen):
        k = pad_len + n + t
        a = np.zeros((1, heads, k if t == 0 else 1, k))
        a[:, :, :, pad_len:] = 1.0 / (n + t)
        steps.append((a,))                      # one layer per step
    return tuple(steps)


def _densities(n_gen, tokens=INPUT_TOKENS):
    """(raw per-token shares, density features) for `n_gen` steps of uniform attention."""
    ratios = sample_category_ratios(
        _uniform_attention_n(n_gen, tokens), 0, 0, len(tokens), n_gen, tokens)
    return ratios, features_from_densities(ratios, region_sizes(tokens))


def test_region_sizes_match_masks():
    assert region_sizes(INPUT_TOKENS) == COUNTS


def test_uniform_attention_gives_neutral_densities():
    """Uniform attention means no preference: every density share is 1/5 and the LL ratio 0.5."""
    _, dens = _densities(3)
    for cat in ("context", "question", "special", "instruct", "output"):
        assert np.isclose(dens[f"dens_{cat}_head_0_layer_0"], 0.2), cat
    assert np.isclose(dens["ll_lookback_head_0_layer_0"], 0.5)


def test_density_shares_sum_to_one():
    d, lr = density_ratios(
        sample_category_ratios(_uniform_attention_n(4), 0, 0, N, 4, INPUT_TOKENS),
        region_sizes(INPUT_TOKENS))
    assert np.allclose(d.sum(axis=-1), 1.0)
    assert ((lr >= 0) & (lr <= 1)).all()
    assert d.shape[0] == 3 and lr.shape[0] == 3        # t=0 skipped (output group is empty)


def test_densities_are_length_invariant():
    """The reason for dividing by group size: with behaviour held fixed (uniform attention), the
    features must not move with prompt or answer length - every density stays at 1/5 and the
    Lookback Lens ratio at 0.5, whatever the passage and answer lengths.

    The raw per-token shares they are built from *do* move - a longer passage takes more of the
    row just by having more tokens - so the last assertion pins down that the invariance comes
    from the length normalization and is not an artefact of the fixture holding everything
    constant.
    """
    seen_ll, seen_dens, seen_share = set(), set(), set()
    for n_sys, n_gen in [(16, 2), (16, 12), (25, 2), (25, 30)]:
        toks = _chat_tokens(n_sys)
        ratios, dens = _densities(n_gen, toks)
        seen_ll.add(round(dens["ll_lookback_head_0_layer_0"], 9))
        for cat in CATEGORIES:
            seen_dens.add((cat, round(dens[f"dens_{cat}_head_0_layer_0"], 9)))
        share_on_prompt = ratios[:, 0, 0, :4].sum(axis=-1).mean()   # before normalization
        seen_share.add(round(float(share_on_prompt), 9))
    assert seen_ll == {0.5}, f"LL ratio drifted with length: {seen_ll}"
    assert seen_dens == {(c, 0.2) for c in CATEGORIES}, f"density drifted with length: {seen_dens}"
    assert len(seen_share) == 4, f"raw share should vary with length, got {seen_share}"


def test_density_responds_to_real_preference():
    """A head that genuinely favours the passage per token must push dens_context above 1/5."""
    n_gen, heads = 3, 2
    steps = []
    for t in range(n_gen):
        k = N + t
        w = np.ones(k)
        w[np.where(region_masks(INPUT_TOKENS)["context"])[0]] = 4.0   # 4x per context token
        a = np.tile(w / w.sum(), (1, heads, k if t == 0 else 1, 1))
        steps.append((a,))
    ratios = sample_category_ratios(tuple(steps), 0, 0, N, n_gen, INPUT_TOKENS)
    dens = features_from_densities(ratios, region_sizes(INPUT_TOKENS))
    assert dens["dens_context_head_0_layer_0"] > 0.2
    assert dens["ll_lookback_head_0_layer_0"] > 0.5


def test_single_token_answer_does_not_divide_by_zero():
    _, dens = _densities(1)
    assert np.isfinite(dens["ll_lookback_head_0_layer_0"])
    assert np.isclose(dens["dens_output_head_0_layer_0"], 0.0)   # nothing generated yet


def test_ll_lookback_matches_paper_formula():
    """`ll_lookback` must equal Chuang et al.'s ratio computed straight off the attention rows.

    Transcribed from their reference implementation (voidism/Lookback-Lens,
    step01_extract_attns.py):

        attn_on_context    = attentions[i][l][0, :, -1, :context_length].mean(-1)
        attn_on_new_tokens = attentions[i][l][0, :, -1, context_length:].mean(-1)
        lookback_ratio     = attn_on_context / (attn_on_context + attn_on_new_tokens)

    with `context_length` = the whole prompt, which is the group convention this repo uses (see
    the README for where that differs from theirs). The other density tests check properties the
    ratio ought to have - uniform attention gives 0.5, length does not move it - which a subtly
    wrong formula could satisfy too. This one pins the definition, so no refactor of
    density_ratios can silently change what is being computed.
    """
    rng = np.random.default_rng(0)
    heads, n_gen = 3, 5
    steps = []
    for t in range(n_gen):                  # random rows, softmax-normalized like real attention
        k = N + t
        w = rng.random((1, heads, k if t == 0 else 1, k))
        steps.append((w / w.sum(-1, keepdims=True),))
    steps = tuple(steps)

    _, lr = density_ratios(
        sample_category_ratios(steps, 0, 0, N, n_gen, INPUT_TOKENS), region_sizes(INPUT_TOKENS))

    for t in range(1, n_gen):               # t=0 is skipped: nothing generated yet to attend to
        row = steps[t][0][0, :, -1, :]
        a_ctx = row[:, :N].sum(1) / N       # attention per prompt token
        a_new = row[:, N:].sum(1) / t       # attention per generated token
        assert np.allclose(lr[t - 1, 0], a_ctx / (a_ctx + a_new)), f"step {t}"


def test_region_masks_shift_invariant():
    """Different system-prompt lengths shift the whole prefix; the passage must still be found.
    Reproduces the 'context starts at 45 not 43' failure and proves dynamic masks handle it."""
    for n_sys in (16, 18, 25):
        toks = _chat_tokens(n_sys)
        assert assert_prompt_layout(toks) is True
        m = region_masks(toks)
        c = find_loc(toks, ("Context", ":"))
        assert set(np.where(m["context"])[0]) == set(range(c + 2, c + 2 + 7))
        assert m["question"].sum() == 5


def test_assert_prompt_layout_rejects_plain_prompt():
    plain = ["Context", ":"] + [f"c{i}" for i in range(10)] + ["Question", ":", "q", "Answer", ":"]
    try:
        assert_prompt_layout(plain)
    except AssertionError:
        pass
    else:
        raise AssertionError("plain prompt (no chat template) should have been rejected")


class _DummyVerifier:
    def __init__(self):
        self.calls = 0

    def judge(self, *a, **k):
        self.calls += 1
        return False


def test_exact_match_short_circuits_api():
    v = _DummyVerifier()
    ok, method = verify(v, "ctx", "q", "Paris", "The answer is paris.")
    assert ok and method == "exact" and v.calls == 0   # containment + normalization, no API


def test_api_used_on_miss():
    v = _DummyVerifier()
    ok, method = verify(v, "ctx", "q", "Paris", "London")
    assert not ok and method == "api" and v.calls == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
