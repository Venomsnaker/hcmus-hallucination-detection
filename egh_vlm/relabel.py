"""Content-aware relabeling of PhD-style hallucination datasets.

The original PhD label is a first-token-match comparator on the model's
yes/no answer, which couples the label to the polarity axis. This module
re-labels each (task, subject, gt, hitem, model_answer) tuple by sending
it to a GPT judge that checks the *content* of the explanation:

  - does the explanation claim the truthful fact `gt`?
  - does the explanation falsely claim the hallucinated fact `hitem`?

The judge does NOT see the image: PhD's `gt` and `hitem` already encode
the ground truth, and the judge only needs to map explanation -> content.

Output schema (per sample):
    claims_gt:     True | False | None  (None = unclear)
    claims_hitem:  True | False | None
    judge_conf:    float in [0, 1]
    label_content: 0 (truthful) | 1 (hallucinated) | -1 (drop)
    judge_reason:  short rationale string

Usage:

    from egh_vlm.relabel import GPTJudge, relabel_dataset, full_analysis
    judge = GPTJudge(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    relabeled = relabel_dataset(dataset, judge, save_path="...json", save_every=50)
    full_analysis(relabeled)  # detailed diagnostic output
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Callable, Iterable

from tqdm import tqdm


__all__ = [
    "GPTJudge",
    "JudgeResult",
    "relabel_dataset",
    "summarize_relabel",
    "analyze_hallucination_sources",
    "analyze_calibration",
    "full_analysis",
    "derive_label",
]


JUDGE_SYSTEM_PROMPT = """You are a careful annotator of vision-language model outputs.

You will be given:
  - the TASK (counting, object, attribute, positional, or sentiment)
  - the SUBJECT being asked about
  - the TRUTHFUL value `gt`  (the correct fact about the image)
  - the HALLUCINATED value `hitem`  (a wrong fact the model might assert)
  - the model's full ANSWER (a yes/no token followed by an explanation)

Your job is to read the ANSWER and decide:
  1. Does the answer's explanation (ignoring the leading yes/no) assert
     that the truthful value `gt` is true of the image?
  2. Does the answer's explanation (ignoring the leading yes/no) assert
     that the hallucinated value `hitem` is true of the image?

Important rules:
  - Judge the EXPLANATION's content, not the leading yes/no token.
  - "Asserts X" means the explanation states X as a fact about the image,
    not as a possibility or a denial.
  - For counting tasks, numerical equivalents count (e.g. "three" == "3").
  - For attribute/positional/sentiment, paraphrases and synonyms count
    (e.g. "domed" == "dome-shaped", "to the left" == "on the left side").
  - If the explanation is generic/evasive and addresses neither value,
    set both to false.

Respond ONLY with a JSON object of the form:
{
  "claims_gt":    true | false | null,
  "claims_hitem": true | false | null,
  "confidence":   <float in [0, 1]>,
  "reason":       "<one short sentence>"
}
Use null only when truly ambiguous. No prose outside the JSON.
"""


JUDGE_USER_TEMPLATE = """TASK: {task}
SUBJECT: {subject}
TRUTHFUL VALUE (gt): {gt}
HALLUCINATED VALUE (hitem): {hitem}
QUESTION: {question}
ANSWER:
\"\"\"
{answer}
\"\"\"

Return the JSON now."""


def _parse_bool(x):
    if isinstance(x, bool) or x is None:
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
        if s in ("null", "none", "unclear", ""):
            return None
    return None


def derive_label(claims_gt, claims_hitem, question_gt: int | None = None) -> int:
    """Map judge output to a content-aware label.
    0 = truthful, 1 = hallucinated, -1 = drop (unclear / off-topic).

    Args:
        claims_gt: Whether explanation asserts the truthful value
        claims_hitem: Whether explanation asserts the hallucinated value
        question_gt: 1 if question asks about gt (truthful=yes), 
                     0 if question asks about hitem (truthful=no)
    """
    if claims_hitem is True:
        return 1  # Explicitly claims hallucination = always bad

    if question_gt == 1:
        # Question asks about gt (e.g., "Is there a cat?"), expects "yes"
        # Must assert gt and deny hitem to be truthful
        if claims_gt is True and claims_hitem is False:
            return 0
        if claims_gt is False and claims_hitem is False:
            # Failed to ground - evasive or wrong denial
            return 1
    elif question_gt == 0:
        # Question asks about hitem (e.g., "Is there a helmet?"), expects "no"
        # Only need to deny hitem to be truthful. Mentioning gt is fine (it's explaining what IS there)
        if claims_hitem is False:
            return 0  # Correctly denies the hallucinated item = truthful answer
        # If claims_hitem is True, already caught above
    else:
        # No polarity info - fallback to original logic
        if claims_gt is True and claims_hitem is False:
            return 0
        if claims_gt is False and claims_hitem is False:
            return 1  # Unknown if evasive is bad without polarity

    return -1  # Unclear cases (e.g., claims_gt is None)


@dataclass
class JudgeResult:
    claims_gt: bool | None
    claims_hitem: bool | None
    confidence: float
    reason: str
    label_content: int
    question_gt: int | None  # polarity used for labeling (if provided)
    raw: str  # the raw judge response text, for debugging


class GPTJudge:
    """Thin wrapper around the OpenAI Chat Completions API.

    Reuses the same pattern as llm/self_check_gpt/selfcheck_prompt_api.py.
    Swap the .judge() body if you want a different provider (Anthropic, vLLM, etc.).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 200,
        retries: int = 3,
        retry_sleep_s: float = 1.5,
    ):
        from openai import OpenAI  # lazy import so the module loads without the dep
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries
        self.retry_sleep_s = retry_sleep_s

    def judge(self, task, subject, gt, hitem, question, answer) -> JudgeResult:
        user_msg = JUDGE_USER_TEMPLATE.format(
            task=task, subject=subject, gt=gt, hitem=hitem,
            question=question or "(not provided)",
            answer=(answer or "").strip(),
        )
        last_err = None
        for attempt in range(self.retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=self.temperature,
                    max_completion_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content
                obj = json.loads(raw)
                cg = _parse_bool(obj.get("claims_gt"))
                ch = _parse_bool(obj.get("claims_hitem"))
                conf = float(obj.get("confidence", 0.0))
                reason = str(obj.get("reason", ""))[:300]
                # Note: question_gt is not passed to judge, so we use derive_label without it
                # The caller (relabel_dataset) will recalculate label with question_gt if available
                return JudgeResult(
                    claims_gt=cg, claims_hitem=ch, confidence=conf,
                    reason=reason, label_content=derive_label(cg, ch), question_gt=None, raw=raw,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_sleep_s * (attempt + 1))
        # All retries failed -> mark as drop with reason.
        return JudgeResult(
            claims_gt=None, claims_hitem=None, confidence=0.0,
            reason=f"judge_error: {last_err}", label_content=-1, question_gt=None, raw="",
        )


def relabel_dataset(
    dataset: Iterable[dict],
    judge: GPTJudge | Callable,
    save_path: str | None = None,
    save_every: int = 100,
    skip_existing: bool = True,
) -> list[dict]:
    """Re-label a PhD-style dataset list-of-dicts in place (returns a new list).

    Each input item must contain: task, subject, gt, hitem, question, answer, id.
    `judge` can be a GPTJudge instance OR any callable with the same .judge() signature.

    Resumable: if save_path exists and skip_existing, items already labeled there
    are kept and not re-judged.
    """
    judge_fn = judge.judge if isinstance(judge, GPTJudge) else judge

    # Resume support
    existing = {}
    if save_path and skip_existing and os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                existing[r["id"]] = r

    out: list[dict] = []
    n_done_since_save = 0
    n_cached = 0
    n_new = 0
    items = list(dataset)
    desc = f"Relabel ({judge.model if isinstance(judge, GPTJudge) else 'judge'})"
    if existing:
        desc += f" | {len(existing)} cached"

    for item in tqdm(items, desc=desc):
        if item["id"] in existing:
            out.append(existing[item["id"]])
            n_cached += 1
            continue
        n_new += 1
        r = judge_fn(
            task=item["task"], subject=item["subject"],
            gt=item["gt"], hitem=item["hitem"],
            question=item.get("question"), answer=item.get("answer"),
        )
        # Recalculate label with question_gt if available
        qgt = item.get("question_gt")
        if qgt is not None:
            label_content = derive_label(r.claims_gt, r.claims_hitem, int(qgt))
        else:
            label_content = r.label_content

        new_item = {
            **item,
            "claims_gt": r.claims_gt,
            "claims_hitem": r.claims_hitem,
            "judge_conf": r.confidence,
            "label_content": label_content,
            "judge_reason": r.reason,
        }
        out.append(new_item)
        n_done_since_save += 1
        if save_path and n_done_since_save >= save_every:
            _atomic_save(out, save_path)
            print(f"  [Checkpoint] Saved {len(out)} samples to {save_path}")
            n_done_since_save = 0

    if save_path:
        _atomic_save(out, save_path)

    print(f"\n[Relabel complete] Total: {len(out)} | Cached: {n_cached} | Newly judged: {n_new}")
    return out


def _atomic_save(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def summarize_relabel(relabeled: list[dict]) -> dict:
    """Quick sanity stats: distribution of new labels and old/new agreement."""
    from collections import Counter
    n = len(relabeled)
    new_counts = Counter(r["label_content"] for r in relabeled)
    # Compare new label to old `label` (the polarity-based one) when present
    agree = 0
    n_compared = 0
    flips = Counter()  # (old, new) -> count
    polarity_split = Counter()  # (question_gt, label_content) -> count
    for r in relabeled:
        if r["label_content"] == -1:
            continue
        old = r.get("label")
        if old is None:
            continue
        n_compared += 1
        agree += int(old == r["label_content"])
        flips[(int(old), int(r["label_content"]))] += 1
        qgt = r.get("question_gt")
        if qgt is not None:
            polarity_split[(int(qgt), int(r["label_content"]))] += 1

    print("=== New label distribution ===")
    for k in sorted(new_counts):
        print(f"  label_content={k:>2}: {new_counts[k]:5d}  ({new_counts[k]/n:.1%})")
    print(f"\n=== Old (polarity) vs new (content) agreement ===")
    if n_compared:
        print(f"  agreement: {agree}/{n_compared} = {agree/n_compared:.1%}")
        for (o, n_), c in sorted(flips.items()):
            tag = "agree" if o == n_ else "FLIP "
            print(f"  {tag} old={o} new={n_}: {c}")
    print(f"\n=== question_gt x new label (should be roughly balanced) ===")
    for (qgt, lc), c in sorted(polarity_split.items()):
        print(f"  question_gt={qgt}  label_content={lc}: {c}")
    return {"new_counts": dict(new_counts), "flips": dict(flips),
            "polarity_split": dict(polarity_split)}


def analyze_hallucination_sources(relabeled: list[dict]) -> dict:
    """Break down why samples are labeled hallucinated (label_content=1).

    Distinguishes between:
      - claims_hitem=True: model explicitly asserts the hallucinated item
      - evasive: claims_gt=False and claims_hitem=False (no clear grounding)
    """
    from collections import Counter

    hallucinated = [r for r in relabeled if r.get("label_content") == 1]
    n = len(hallucinated)
    if n == 0:
        print("\n=== Hallucination source analysis ===")
        print("  No hallucinated samples found.")
        return {}

    sources = Counter()
    task_breakdown = Counter()
    confidence_by_source = {"explicit_hitem": [], "evasive": []}

    for r in hallucinated:
        ch = r.get("claims_hitem")
        cg = r.get("claims_gt")
        conf = r.get("judge_conf", 0)
        task = r.get("task", "unknown")

        if ch is True:
            sources["explicit_hitem"] += 1
            confidence_by_source["explicit_hitem"].append(conf)
            task_breakdown[(task, "explicit_hitem")] += 1
        elif cg is False and ch is False:
            sources["evasive"] += 1
            confidence_by_source["evasive"].append(conf)
            task_breakdown[(task, "evasive")] += 1
        else:
            sources[f"other (cg={cg}, ch={ch})"] += 1

    print("\n=== Hallucination source analysis ===")
    print(f"  Total hallucinated: {n}")
    for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
        pct = cnt / n * 100
        confs = confidence_by_source.get(src, [])
        avg_conf = sum(confs) / len(confs) if confs else 0
        print(f"  {src:20s}: {cnt:4d} ({pct:5.1f}%)  avg_conf={avg_conf:.2f}")

    print("\n  By task:")
    for (task, src), cnt in sorted(task_breakdown.items(), key=lambda x: -x[1]):
        print(f"    {task:12s} {src:15s}: {cnt}")

    return {
        "sources": dict(sources),
        "task_breakdown": {f"{t}_{s}": c for (t, s), c in task_breakdown.items()},
        "confidence_by_source": {k: sum(v)/len(v) if v else 0
                                  for k, v in confidence_by_source.items()},
    }


def analyze_calibration(relabeled: list[dict]) -> dict:
    """Check if judge confidence correlates with correctness.

    For samples with known old labels, compare confidence when agreeing vs disagreeing.
    """
    agree_confs = []
    disagree_confs = []

    for r in relabeled:
        old = r.get("label")
        new = r.get("label_content")
        conf = r.get("judge_conf", 0)
        if old is None or new is None or new == -1:
            continue
        if old == new:
            agree_confs.append(conf)
        else:
            disagree_confs.append(conf)

    if not agree_confs or not disagree_confs:
        print("\n=== Calibration analysis ===")
        print("  Insufficient data (need both agreements and disagreements)")
        return {}

    import statistics
    agree_mean = statistics.mean(agree_confs)
    disagree_mean = statistics.mean(disagree_confs)

    print("\n=== Judge calibration (confidence analysis) ===")
    print(f"  When judge agrees with old label  : n={len(agree_confs)}, conf={agree_mean:.3f}")
    print(f"  When judge disagrees (flips label): n={len(disagree_confs)}, conf={disagree_mean:.3f}")
    if agree_mean > disagree_mean:
        print(f"  -> Well-calibrated: higher confidence on agreements")
    else:
        print(f"  -> WARNING: higher confidence on disagreements (overconfident when wrong)")

    # Binned analysis
    print("\n  Confidence distribution:")
    bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]
    for lo, hi in bins:
        agree_bin = sum(1 for c in agree_confs if lo <= c < hi)
        disagree_bin = sum(1 for c in disagree_confs if lo <= c < hi)
        total = agree_bin + disagree_bin
        if total > 0:
            accuracy = agree_bin / total
            print(f"    conf [{lo:.2f}, {hi:.2f}): {total:3d} samples, accuracy={accuracy:.1%}")

    return {
        "agree_mean_conf": agree_mean,
        "disagree_mean_conf": disagree_mean,
        "agree_count": len(agree_confs),
        "disagree_count": len(disagree_confs),
    }


def full_analysis(relabeled: list[dict]) -> dict:
    """Run all analyses and return combined report."""
    summary = summarize_relabel(relabeled)
    hallucination_sources = analyze_hallucination_sources(relabeled)
    calibration = analyze_calibration(relabeled)

    # Additional steering-specific check
    print("\n=== Steering suitability check ===")
    from collections import Counter
    task_split = Counter((r.get("task"), r.get("label_content")) for r in relabeled)
    print("  Label distribution by task:")
    for (task, label), cnt in sorted(task_split.items()):
        print(f"    {task:12s} label={label}: {cnt:3d}")

    # Check for severe polarity skew
    polarity_skew = summary.get("polarity_split", {})
    total_0 = sum(v for k, v in polarity_skew.items() if k[0] == 0)
    total_1 = sum(v for k, v in polarity_skew.items() if k[0] == 1)
    truthful_0 = polarity_skew.get((0, 0), 0)
    truthful_1 = polarity_skew.get((1, 0), 0)

    if total_0 > 0 and total_1 > 0:
        rate_0 = truthful_0 / total_0
        rate_1 = truthful_1 / total_1
        print(f"\n  Truthful rate by question_gt:")
        print(f"    question_gt=0 (neg): {rate_0:.1%} ({truthful_0}/{total_0})")
        print(f"    question_gt=1 (pos): {rate_1:.1%} ({truthful_1}/{total_1})")
        if abs(rate_0 - rate_1) > 0.15:
            print(f"  -> WARNING: {abs(rate_0 - rate_1):.1%} polarity skew detected!")
            print(f"     Consider stratified sampling or rebalancing for steering.")
        else:
            print(f"  -> OK: balanced polarity (<15% difference)")

    return {
        "summary": summary,
        "hallucination_sources": hallucination_sources,
        "calibration": calibration,
    }
