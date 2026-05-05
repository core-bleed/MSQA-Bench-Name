"""
Create an automatic baseline annotation for gold_set_sample.csv.

Evaluates each Q/A/context triple on four dimensions:
  - answer_correct:   Yes / Partial / No
  - evidence_support:  Yes / Partial / No
  - evidence_quality:  Good / Weak / Missing
  - question_clarity:  Good / Ambiguous / Bad

Uses text-overlap heuristics (token, n-gram, number matching)
plus structural checks (garbled text, truncation, question form). This helper
is not the human gold audit reported in the paper.
"""

import csv
import re
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────

STOP_WORDS = frozenset(
    "the and was were are for that this with from has had have been not but "
    "they its also than can may which when what how who where why does did "
    "will would could should about their these those some such other into "
    "more most only between after before during each both "
    "text according".split()
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def word_tokens(text: str) -> list[str]:
    """Meaningful tokens (len >= 2, no stop words).  Keep abbreviations."""
    words = re.findall(r"\b[a-zA-Z\u0400-\u04FF\u00C0-\u024F]{2,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


def tech_tokens(text: str) -> set[str]:
    """Extract technical abbreviations (uppercase 2+ chars, hyphenated terms)."""
    abbrevs = set(re.findall(r"\b[A-Z][A-Z0-9/-]{1,}\b", text))
    hyphenated = set(re.findall(r"\b\w+-\w+(?:-\w+)*\b", text.lower()))
    return abbrevs | hyphenated


def extract_numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+\.?\d*", text))


def ngrams(text: str, n: int = 4) -> set[str]:
    words = text.split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def token_overlap(answer_tokens: list[str], context_tokens: set[str]) -> float:
    if not answer_tokens:
        return 0.0
    return sum(1 for t in answer_tokens if t in context_tokens) / len(answer_tokens)


def number_overlap(answer: str, context: str) -> float:
    a_nums = extract_numbers(answer)
    c_nums = extract_numbers(context)
    if not a_nums:
        return 1.0
    return len(a_nums & c_nums) / len(a_nums)


def phrase_overlap(answer: str, context: str, n: int = 3) -> float:
    a_words = normalize(answer).split()
    phrases = [" ".join(a_words[i : i + n]) for i in range(len(a_words) - n + 1)]
    if not phrases:
        return 0.0
    ctx_norm = normalize(context)
    return sum(1 for p in phrases if p in ctx_norm) / len(phrases)


def is_garbled(text: str) -> bool:
    if not text or len(text) < 20:
        return True
    alpha = sum(c.isalpha() for c in text)
    alnum = sum(c.isalnum() for c in text)
    total = len(text)
    # truly garbled: very few alphabetic chars AND few alphanumerics
    if alpha / total < 0.15:
        return True
    # table data with lots of numbers is NOT garbled
    if alnum / total > 0.45:
        return False
    # low alpha but still some readable text
    return alpha / total < 0.20


def is_hedging(answer: str) -> bool:
    phrases = [
        "does not provide", "does not specify", "does not mention",
        "not mentioned", "not stated", "not specified", "not discussed",
        "text does not", "unclear from", "no information",
        "cannot be determined", "not explicitly", "it simply states",
        "does not explain", "no reason",
    ]
    a = answer.lower()
    return any(p in a for p in phrases)


def is_non_english(text: str) -> bool:
    if re.search(r"[\u0400-\u04FF]{5,}", text):
        return True
    pt_es = re.compile(
        r"\b(?:são|não|também|após|foram|para|como|está|através|entre|"
        r"permitir|obtenção|informação|peptídeo|amostra|quantitativa)\b", re.I
    )
    return bool(pt_es.search(text))


# ── annotators ───────────────────────────────────────────────────────

def substring_match(answer: str, context: str, min_len: int = 8) -> float:
    """Check what fraction of the answer appears as substrings in context."""
    a = normalize(answer)
    c = normalize(context)
    # try sliding windows of decreasing size
    words = a.split()
    if len(words) < 3:
        # for very short answers, do direct substring check
        return 1.0 if a in c else 0.0
    matched_words = 0
    for w_len in range(min(len(words), 8), 1, -1):
        for i in range(len(words) - w_len + 1):
            chunk = " ".join(words[i : i + w_len])
            if chunk in c:
                matched_words = max(matched_words, w_len)
                return matched_words / len(words)
    return 0.0


def tech_overlap(answer: str, context: str) -> float:
    """Overlap of technical abbreviations between answer and context."""
    a_tech = tech_tokens(answer)
    c_tech = tech_tokens(context)
    if not a_tech:
        return 1.0
    return len(a_tech & c_tech) / len(a_tech)


def annotate_answer_correct(question: str, answer: str, context: str,
                            qtype: str, qscore: float) -> str:
    if is_garbled(context):
        return "No"

    a_tok = word_tokens(answer)
    c_tok_set = set(word_tokens(context))
    t_overlap = token_overlap(a_tok, c_tok_set)
    n_overlap = number_overlap(answer, context)
    p_overlap = phrase_overlap(answer, context)
    sub_match = substring_match(answer, context)
    tch_overlap = tech_overlap(answer, context)

    ng_overlap_ratio = 0.0
    a_ng = ngrams(normalize(answer))
    c_ng = ngrams(normalize(context))
    if a_ng:
        ng_overlap_ratio = len(a_ng & c_ng) / len(a_ng)

    # short answers (< 6 content tokens): rely on substring + tech match
    short_answer = len(a_tok) < 6

    if is_hedging(answer):
        return "Yes" if t_overlap > 0.10 else "Partial"

    # ── Yes ──
    if sub_match > 0.35:
        return "Yes"
    if ng_overlap_ratio > 0.20 and n_overlap >= 0.5:
        return "Yes"
    if t_overlap > 0.45 and n_overlap >= 0.5:
        return "Yes"
    if p_overlap > 0.30 and n_overlap >= 0.5:
        return "Yes"
    if t_overlap > 0.40:
        return "Yes"
    if short_answer and tch_overlap >= 0.5 and n_overlap >= 0.5:
        return "Yes"
    if short_answer and sub_match > 0.0 and n_overlap >= 0.5:
        return "Yes"
    # high quality score + decent overlap → trust it
    if qscore >= 0.82 and t_overlap > 0.25 and n_overlap >= 0.3:
        return "Yes"

    # ── Partial ──
    if t_overlap > 0.20 and n_overlap >= 0.3:
        return "Partial"
    if ng_overlap_ratio > 0.08:
        return "Partial"
    if t_overlap > 0.18:
        return "Partial"
    if p_overlap > 0.10:
        return "Partial"
    if n_overlap > 0.4 and t_overlap > 0.10:
        return "Partial"
    if short_answer and (tch_overlap > 0.3 or n_overlap > 0.3):
        return "Partial"
    if qscore >= 0.80 and t_overlap > 0.10:
        return "Partial"

    return "No"


def annotate_evidence_support(question: str, answer: str, context: str,
                              qtype: str) -> str:
    if is_garbled(context):
        return "No"

    a_tok = word_tokens(answer)
    c_tok_set = set(word_tokens(context))
    t_overlap = token_overlap(a_tok, c_tok_set)
    n_overlap = number_overlap(answer, context)
    p_overlap = phrase_overlap(answer, context)
    sub_match = substring_match(answer, context)
    tch_overlap = tech_overlap(answer, context)
    short_answer = len(a_tok) < 6

    if is_hedging(answer):
        return "Yes"

    # Yes
    if sub_match > 0.30:
        return "Yes"
    if p_overlap > 0.25 and n_overlap >= 0.5:
        return "Yes"
    if t_overlap > 0.40 and n_overlap >= 0.4:
        return "Yes"
    if t_overlap > 0.38:
        return "Yes"
    if short_answer and (tch_overlap >= 0.5 or sub_match > 0.0):
        return "Yes"

    # Partial
    if t_overlap > 0.18 or p_overlap > 0.10:
        return "Partial"
    if n_overlap > 0.4 and t_overlap > 0.08:
        return "Partial"
    if short_answer and (tch_overlap > 0.2 or n_overlap > 0.3):
        return "Partial"
    if t_overlap > 0.10:
        return "Partial"

    return "No"


def annotate_evidence_quality(question: str, context: str, qtype: str) -> str:
    if is_garbled(context):
        return "Missing"
    if len(context) < 50:
        return "Missing"

    q_tok = set(word_tokens(question))
    c_tok = set(word_tokens(context))
    q_overlap = len(q_tok & c_tok) / max(len(q_tok), 1)
    alpha_ratio = sum(c.isalpha() for c in context) / max(len(context), 1)
    num_density = sum(c.isdigit() for c in context) / max(len(context), 1)

    # mostly table / numeric dump
    if num_density > 0.25 and alpha_ratio < 0.40:
        return "Weak"

    if alpha_ratio > 0.45 and q_overlap > 0.25 and len(context) > 100:
        return "Good"
    if q_overlap > 0.35 and len(context) > 120:
        return "Good"

    if q_overlap < 0.10:
        return "Weak"
    if alpha_ratio < 0.35:
        return "Weak"
    if len(context) < 80:
        return "Weak"

    if q_overlap > 0.15:
        return "Good"

    return "Weak"


def annotate_question_clarity(question: str, qtype: str) -> str:
    q_words = question.split()
    q_len = len(q_words)
    q_low = question.lower()

    if q_len < 4:
        return "Bad"

    # non-English: accept if reasonably formed
    if is_non_english(question):
        return "Good" if q_len >= 5 else "Ambiguous"

    wh = q_low.startswith(("what", "how", "why", "which", "when", "where", "who"))
    verb = q_low.startswith(("is", "are", "was", "were", "do", "does", "did", "can", "could"))
    according = "according to" in q_low
    is_q = wh or verb or according or question.strip().endswith("?")

    # detect named entities: skip leading question words
    q_body = re.sub(r"^(?:What|How|Why|Which|When|Where|Who|Is|Are|Was|Were|"
                    r"Do|Does|Did|Can|Could|According)\b\s*", "", question)
    has_entity = bool(re.search(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*|[A-Z]{2,}|\d+", q_body))
    vague_patterns = [
        "how effective", "what happens", "describe the process",
        "what are the results", "what was found", "what is the result",
        "how does the text describe",
    ]
    vague = any(p in q_low for p in vague_patterns)

    # self-referential ("according to the text") without specific scope
    self_ref = any(p in q_low for p in ["according to the text", "the text describe"])

    if not is_q:
        return "Bad"
    if vague and not has_entity:
        return "Ambiguous"
    if self_ref and not has_entity:
        return "Ambiguous"
    if q_len >= 6 and has_entity:
        return "Good"
    if q_len >= 5:
        return "Good"

    return "Ambiguous"


# ── main ─────────────────────────────────────────────────────────────

def annotate_row(row: dict) -> dict:
    q = row["question"]
    a = row["answer"]
    c = row["context"]
    qtype = row.get("question_type", "unknown")
    qscore = float(row.get("quality_score", 0))

    row["answer_correct"]   = annotate_answer_correct(q, a, c, qtype, qscore)
    row["evidence_support"]  = annotate_evidence_support(q, a, c, qtype)
    row["evidence_quality"]  = annotate_evidence_quality(q, c, qtype)
    row["question_clarity"]  = annotate_question_clarity(q, qtype)
    row["annotator_id"]      = "auto_overlap_baseline"
    row["notes"]             = ""

    # ── consistency checks ──
    # if evidence is missing, answer can't be verified as correct
    if row["evidence_quality"] == "Missing" and row["answer_correct"] == "Yes":
        row["answer_correct"] = "Partial"
    # if answer is wrong, evidence support can't be full
    if row["answer_correct"] == "No" and row["evidence_support"] == "Yes":
        row["evidence_support"] = "Partial"

    # add specific notes for edge cases
    notes = []
    if is_garbled(c):
        notes.append("garbled_context")
    if is_non_english(q) or is_non_english(a):
        notes.append("non_english")
    if is_hedging(a):
        notes.append("hedging_answer")
    if len(c) < 100:
        notes.append("short_context")
    row["notes"] = "; ".join(notes)

    return row


def main():
    src = Path("paper_results/annotation/gold_set_sample.csv")
    dst = Path("paper_results/annotation/gold_set_annotated.csv")

    fieldnames = [
        "annotation_id", "question", "answer", "context",
        "question_type", "quality_score",
        "answer_correct", "evidence_support", "evidence_quality",
        "question_clarity", "annotator_id", "notes",
    ]

    rows = []
    with src.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(annotate_row(row))

    with dst.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    # print summary
    from collections import Counter
    print(f"Annotated {len(rows)} records → {dst}\n")

    for dim in ("answer_correct", "evidence_support", "evidence_quality", "question_clarity"):
        counts = Counter(r[dim] for r in rows)
        print(f"{dim}:")
        for label, n in sorted(counts.items()):
            print(f"  {label:12s}  {n:3d}  ({n/len(rows)*100:5.1f}%)")
        print()

    # notes distribution
    note_counts = Counter()
    for r in rows:
        for n in r["notes"].split("; "):
            if n:
                note_counts[n] += 1
    if note_counts:
        print("notes flags:")
        for flag, n in note_counts.most_common():
            print(f"  {flag:20s}  {n}")


if __name__ == "__main__":
    main()
