"""
Attention feature extraction (v2): per-token attention densities, following Lookback Lens.

For each generated (output) token and each (layer, head), we split that token's attention
over ALL preceding keys into five categories:

    context, question, special, instruct   (the four prompt token-types)
    output                                  (previously generated tokens - the "new" term)

Each category's attention is divided by the number of tokens in it, giving an attention
*density* per token, and the five densities are renormalized to sum to 1. This is how Lookback
Lens (Chuang et al., 2024) defines its ratio: dividing by the group size is what makes the
feature independent of how long the passage and the answer are, so a head that spreads attention
evenly scores the same whatever the lengths. ``dens_*`` applies that to all five categories;
``ll_lookback_*`` is their two-group statistic, the whole prompt against the model's own output -
their own code puts the response marker in the *new* group instead, see the README.

The per-(layer, head) feature is the mean over the response tokens. Lookback Lens averages over a
span instead (a sentence, or a sliding window of 8 tokens), since it predicts per span.

Pipeline:
  - step 02 has the raw attentions and calls ``sample_category_ratios`` to dump a small
    ``cat_ratios`` array of shape ``(n_output_tokens, n_layers, n_heads, 5)`` per sample. Those
    are per-token shares of the attention row - the raw split, before any length normalization.
  - step 03 calls ``features_from_densities`` to normalize and mean over output tokens -> wide
    features ``dens_*`` and ``ll_lookback_*``.

Densities are derived from the dumped shares rather than the raw attention maps: the row total is
a common factor and cancels when the densities are renormalized, so nothing downstream needs a
GPU rerun of step 02.

At modelling time ``output`` (attention to the model's own generated tokens) is dropped as the
reference category, leaving four densities and avoiding the dummy-variable trap.
"""
import numpy as np

CATEGORIES = ['context', 'question', 'special', 'instruct', 'output']
_EPS = 1e-12
_CTRL = ('system', 'user', 'assistant')   # chat role tokens (besides <|im_start|>/<|im_end|>)


def find_loc(token_list, find_tup):
    """Index of the first occurrence of the tuple ``find_tup`` in ``token_list``."""
    for i in range(len(token_list) - len(find_tup) + 1):
        if find_tup == tuple(token_list[i:len(find_tup) + i]):
            return i
    return None


def region_masks(input_tokens):
    """Boolean masks over the input positions for the four prompt token-types.

    Detected dynamically from the chat-template markers, so it is robust to prefix-length
    changes across tokenizer/transformers versions (no fixed offsets):

      context  : the passage between ``Context :`` and ``Question``
      question : from ``Question`` to the end of the user turn (the closing ``<|im_end|>``)
      special  : chat-control scaffolding (``<|im_start|>``/``<|im_end|>``, role names, the
                 trailing generation prompt)
      instruct : everything else (system message + the fixed instruction + the marker words)
    """
    toks = [str(t) for t in input_tokens]
    n = len(toks)
    masks = {k: np.zeros(n, dtype=bool) for k in ('context', 'question', 'special', 'instruct')}

    c = find_loc(toks, ('Context', ':'))
    q = find_loc(toks, ('Question', ':'))

    for i, t in enumerate(toks):                       # chat-control tokens -> special
        if ('im_start' in t) or ('im_end' in t) or t in _CTRL:
            masks['special'][i] = True

    if c is not None:                                  # context passage (skip the 'Context :' marker)
        masks['context'][c + 2:(q if q is not None else n)] = True
    if q is not None:                                  # question up to the closing <|im_end|>
        end = q
        while end < n and 'im_end' not in toks[end]:
            end += 1
        masks['question'][q:end] = True

    masks['instruct'][~(masks['context'] | masks['question'] | masks['special'])] = True
    return masks


def assert_prompt_layout(input_tokens):
    """Fail loudly if the prompt isn't chat-templated or the Context/Question markers are missing.

    Cheap structural check meant to run on the first sample *before* an expensive generation run,
    so a template/tokenizer mismatch aborts in seconds instead of after the whole GPU job.
    """
    toks = [str(t) for t in input_tokens]
    assert any("im_start" in t for t in toks), (
        "no chat-template tokens (<|im_start|>) found - generate with tokenizer.apply_chat_template")
    c = find_loc(toks, ("Context", ":"))
    q = find_loc(toks, ("Question", ":"))
    assert c is not None, "'Context :' marker not found (tokenizer split it differently)"
    assert q is not None, "'Question :' marker not found (tokenizer split it differently)"
    assert c + 2 < q, "empty context region between 'Context :' and 'Question'"
    m = region_masks(toks)
    assert m["context"].any() and m["question"].any(), "context/question region is empty"
    return True


def _to_np(x):
    """Accept a torch tensor or a numpy array."""
    try:
        return x.detach().float().cpu().numpy()
    except AttributeError:
        return np.asarray(x, dtype=np.float64)


def sample_category_ratios(attentions, b, pad_len, input_len, n_gen, input_tokens):
    """Per-output-token category ratios for sample ``b`` of a (possibly left-padded) batch.

    Args:
        attentions: ``generate`` attentions tuple; ``attentions[t]`` is a tuple over layers
            of tensors ``(batch, heads, query_len, key_len)`` (HuggingFace format).
        b: sample index within the batch.
        pad_len: number of left-pad tokens before sample ``b``'s real prompt (0 if unpadded).
        input_len: real prompt length for sample ``b``.
        n_gen: number of real generated tokens for sample ``b``.
        input_tokens: real prompt token strings (for token-type segmentation).

    Returns:
        Array ``(n_gen, n_layers, n_heads, 5)`` - per output token/layer/head, the ratio of
        attention on ``[context, question, special, instruct, output]`` (rows sum to ~1).
    """
    masks = region_masks(input_tokens)
    prompt_end = pad_len + input_len               # padded index where the prompt ends
    steps = []
    for t in range(n_gen):
        layer_cats = []
        for layer_attn in attentions[t]:           # (batch, heads, query, key)
            a = _to_np(layer_attn[b])              # (heads, query, key)
            assert a.shape[-1] == prompt_end + t, (
                f"attention key length {a.shape[-1]} != expected {prompt_end + t} at step {t} "
                "(padded prompt + t generated keys) - generate/cache layout differs from assumed")
            row = a[:, -1, :]                      # last query row = current token; robust to use_cache
            total = row.sum(axis=1) + _EPS         # (heads,)

            prompt_row = row[:, pad_len:prompt_end]     # attention on the real prompt
            gen_row = row[:, prompt_end:]               # attention on generated tokens so far
            cats = np.stack([
                prompt_row[:, masks['context']].sum(axis=1),
                prompt_row[:, masks['question']].sum(axis=1),
                prompt_row[:, masks['special']].sum(axis=1),
                prompt_row[:, masks['instruct']].sum(axis=1),
                gen_row.sum(axis=1) if gen_row.shape[1] else np.zeros(row.shape[0]),
            ], axis=1)                             # (heads, 5)
            layer_cats.append(cats / total[:, None])
        steps.append(np.stack(layer_cats, axis=0))  # (n_layers, heads, 5)
    return np.stack(steps, axis=0)                  # (n_gen, n_layers, heads, 5)


def per_token_category_ratios(attentions, input_len, input_tokens):
    """Unbatched, unpadded convenience wrapper over :func:`sample_category_ratios`."""
    return sample_category_ratios(attentions, 0, 0, input_len, len(attentions), input_tokens)


def region_sizes(input_tokens):
    """Number of prompt tokens in each of the four input categories."""
    return {k: int(v.sum()) for k, v in region_masks(input_tokens).items()}


def density_ratios(ratios, sizes):
    """Length-normalized category ratios, the way Lookback Lens defines them.

    ``ratios`` holds *shares* of the attention row, ``r_c(t) = S_c(t) / row_total(t)`` - the raw
    split dumped by step 02. A share on its own moves with length: a longer passage collects more
    of the row simply by having more tokens, even when the head's behaviour is unchanged. Lookback
    Lens divides each group's attention by the number of tokens in that group, giving an attention
    *density* per token, and takes the ratio between groups:

        A_c(t) = S_c(t) / n_c            d_c(t) = A_c(t) / sum_c' A_c'(t)

    ``row_total`` is a common factor of every ``A_c`` and cancels in the second step, so the
    shares already dumped by step 02 are enough - the raw attention maps are not needed again.

    ``t = 0`` has generated nothing yet, so the ``output`` group is empty and its density is
    undefined. Lookback Lens never meets this case (their "new" group starts non-empty, holding
    the response marker), so that step is skipped here - unless the answer is a single token and
    there is nothing else to average, where the output density is 0 by construction.

    Args:
        ratios: ``(n_out, L, H, 5)`` per-token category shares, in ``CATEGORIES`` order.
        sizes: ``{category: n_tokens}`` for the four input categories (see ``region_sizes``).

    Returns:
        ``(d, lr)``. ``d`` is ``(n_valid, L, H, 5)``, the five densities normalized to sum to 1.
        ``lr`` is ``(n_valid, L, H)``, the two-group lookback ratio with the whole prompt as one
        group and the generated tokens as the other. The per-token formula is Chuang et al.'s -
        ``tests/test_features.py`` pins it against a transcription of their reference code - but
        the group boundary is this project's: theirs puts the response marker in the new group.
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    n_out = ratios.shape[0]
    # max(., 1) guards an empty region; its share is 0 too, so the density stays 0.
    n_in = np.array([max(sizes[c], 1) for c in CATEGORIES[:4]], dtype=np.float64)

    start = 1 if n_out > 1 else 0
    r = ratios[start:]
    n_gen_so_far = np.maximum(np.arange(start, n_out, dtype=np.float64), 1.0)[:, None, None]

    dens = np.empty_like(r)
    dens[..., :4] = r[..., :4] / n_in                  # per-token density in each prompt region
    dens[..., 4] = r[..., 4] / n_gen_so_far            # per-token density over own output
    d = dens / (dens.sum(axis=-1, keepdims=True) + _EPS)

    a_in = r[..., :4].sum(axis=-1) / n_in.sum()        # whole prompt as one group
    a_out = r[..., 4] / n_gen_so_far
    lr = a_in / (a_in + a_out + _EPS)
    return d, lr


def features_from_densities(ratios, sizes):
    """Mean over response tokens of :func:`density_ratios` -> ``{feature_name: value}``.

    Emits ``dens_{category}_head_{h}_layer_{l}`` (the five normalized densities) and
    ``ll_lookback_head_{h}_layer_{l}`` (the two-group Lookback Lens ratio).
    """
    d, lr = density_ratios(ratios, sizes)
    dm, lrm = d.mean(axis=0), lr.mean(axis=0)
    n_layers, n_heads = lrm.shape
    feats = {}
    for l in range(n_layers):
        for h in range(n_heads):
            for ci, cat in enumerate(CATEGORIES):
                feats[f"dens_{cat}_head_{h}_layer_{l}"] = float(dm[l, h, ci])
            feats[f"ll_lookback_head_{h}_layer_{l}"] = float(lrm[l, h])
    return feats
