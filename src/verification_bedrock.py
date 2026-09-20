"""
Answer verification: exact match first, Amazon Bedrock (Nova Lite) only on a miss.

`verify()` returns (is_correct, method):
  - method="exact": the normalized official answer matched the generated answer (no API call).
  - method="api":   exact match failed, so a Bedrock model judged semantic equivalence.

This keeps cost/latency low - the API is only hit for the answers exact match can't settle.
"""
import re
import boto3


def _normalize(s):
    """SQuAD-style normalization: lowercase, strip punctuation and articles."""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def exact_match(answer, generated_answer):
    """True if the (short) official answer equals or appears in the generated answer."""
    a, g = _normalize(answer), _normalize(generated_answer)
    if not a or not g:
        return False
    return a == g or a in g


class BedrockVerifier:
    """Semantic-equivalence judge via the Bedrock Converse API."""

    SYSTEM = (
        "You judge whether a model's generated answer is correct, i.e. semantically matches "
        "the official answer for the given question. Wording may differ; they only need to "
        "mean the same thing. Reply with exactly one word: true or false."
    )

    def __init__(self, model_id, region):
        self.client = boto3.client("bedrock-runtime", region_name=region)
        self.model_id = model_id

    def judge(self, context, question, answer, generated_answer):
        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Official answer: {answer}\n"
            f"Generated answer: {generated_answer}\n"
            "Is the generated answer correct? Reply true or false."
        )
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": self.SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 5, "temperature": 0.0},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        return "true" in text.lower()


def verify(verifier, context, question, answer, generated_answer):
    """Exact match short-circuits to correct; otherwise ask the Bedrock verifier."""
    if exact_match(answer, generated_answer):
        return True, "exact"
    return verifier.judge(context, question, answer, generated_answer), "api"
