"""Generate retrieval-results figures for the MS-GPT thesis.

Produces three PNGs in ``thesis/figures/``:

* ``retrieval_recall_bar.png`` -- grouped Recall@k bar chart across BM25
  and six fine-tuned dense encoders.
* ``retrieval_radar.png`` -- radar plot of the five retrieval metrics
  across the six fine-tuned dense encoders.
* ``retrieval_improvement_heatmap.png`` -- absolute gain
  (Fine-tuned -- Base) heat-map on R@10, MRR@10, NDCG@10 for the five
  dense encoders for which base numbers are recorded in the paper.

All numbers are read from the repository's result JSONs where possible.
Base-model scores are sourced from each model's
``training_summary.json`` (``metrics_history[0].score``); fine-tuned
scores come from ``eval_results_test.json``; the MiniLM entries use
``training_summary.json`` at the repo root; BM25 comes from
``paper_results/evaluation/bm25_baseline_results.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("thesis_figures")

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "thesis" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Models reported in the MSQA-Bench paper (E5 / BGE / Nomic).
EMB_MODELS: list[tuple[str, Path]] = [
    ("E5-base",    REPO / "models" / "fine_tuned_embeddings_e5_base_v2"),
    ("E5-large",   REPO / "models" / "fine_tuned_embeddings_e5_large_v2"),
    ("BGE-base",   REPO / "models" / "fine_tuned_embeddings_bge_base_en_v1.5"),
    ("BGE-large",  REPO / "models" / "fine_tuned_embeddings_bge_large_en_v1.5"),
    ("Nomic-v1.5", REPO / "models" / "fine_tuned_embeddings_nomic_embed_v1.5"),
]

METRICS = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
COSINE_KEYS = {
    "recall@1":  "qa_retrieval_cosine_recall@1",
    "recall@5":  "qa_retrieval_cosine_recall@5",
    "recall@10": "qa_retrieval_cosine_recall@10",
    "mrr@10":    "qa_retrieval_cosine_mrr@10",
    "ndcg@10":   "qa_retrieval_cosine_ndcg@10",
}


def _load_json(p: Path) -> dict | None:
    try:
        with p.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read %s (%s)", p, exc)
        return None


def collect() -> dict:
    """Return {'bm25': {m: v}, 'ft': {model: {m: v}}, 'base': {model: {m: v}}}."""
    bm25 = _load_json(REPO / "paper_results" / "evaluation" / "bm25_baseline_results.json") or {}

    ft: dict[str, dict[str, float]] = {}
    base: dict[str, dict[str, float]] = {}

    for name, folder in EMB_MODELS:
        ft_json = _load_json(folder / "eval_results_test.json")
        if ft_json:
            ft[name] = {m: float(ft_json[m]) for m in METRICS if m in ft_json}
        ts = _load_json(folder / "training_summary.json")
        if ts and ts.get("metrics_history"):
            score = ts["metrics_history"][0].get("score", {})
            base[name] = {m: float(score[COSINE_KEYS[m]]) for m in METRICS if COSINE_KEYS[m] in score}

    return {"bm25": bm25, "ft": ft, "base": base}


def fig_recall_bar(data: dict) -> Path:
    models = ["BM25"] + list(data["ft"].keys())
    xs = np.arange(len(models))
    width = 0.26

    recall1, recall5, recall10 = [], [], []
    for m in models:
        src = data["bm25"] if m == "BM25" else data["ft"][m]
        recall1.append(src.get("recall@1", np.nan))
        recall5.append(src.get("recall@5", np.nan))
        recall10.append(src.get("recall@10", np.nan))

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(xs - width, recall1,  width, label=r"Recall@1",  color="#4c78a8")
    ax.bar(xs,         recall5,  width, label=r"Recall@5",  color="#f58518")
    ax.bar(xs + width, recall10, width, label=r"Recall@10", color="#54a24b")
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Recall")
    ax.set_title("Retrieval Recall@$k$ on MSQA-Bench (5{,}000 test queries)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "retrieval_recall_bar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_radar(data: dict) -> Path:
    models = list(data["ft"].keys())
    labels = ["R@1", "R@5", "R@10", "MRR@10", "NDCG@10"]
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756",
              "#72b7b2", "#b279a2"]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    for i, m in enumerate(models):
        vals = [data["ft"][m].get(mk, 0.0) for mk in METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, label=m, color=colors[i % len(colors)], linewidth=1.8)
        ax.fill(angles, vals, color=colors[i % len(colors)], alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0.7, 1.0)
    ax.set_rlabel_position(10)
    ax.grid(alpha=0.4)
    ax.set_title("Fine-tuned Embedding Metrics (MSQA-Bench test)", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "retrieval_radar.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_improvement_heatmap(data: dict) -> Path:
    # Only models that have both base and FT, aligned on recall/mrr/ndcg @10.
    metrics = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    mlabels = ["R@1", "R@5", "R@10", "MRR@10", "NDCG@10"]
    models = [m for m in data["ft"] if m in data["base"]]
    mat = np.full((len(models), len(metrics)), np.nan)
    for i, m in enumerate(models):
        for j, mk in enumerate(metrics):
            if mk in data["ft"][m] and mk in data["base"][m]:
                mat[i, j] = data["ft"][m][mk] - data["base"][m][mk]

    fig, ax = plt.subplots(figsize=(7.5, 0.6 + 0.55 * len(models)))
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0.0, vmax=max(0.25, np.nanmax(mat)))
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(mlabels)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(metrics)):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", color="grey", fontsize=9)
            else:
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        color="black" if v < 0.18 else "white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r"Fine-tuned $-$ Base")
    ax.set_title("Absolute Gain from Domain Fine-Tuning")
    fig.tight_layout()
    out = FIG_DIR / "retrieval_improvement_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_publication_year_histogram() -> Path | None:
    """Per-document publication year histogram from enriched.jsonl."""
    src = REPO / "paper_results" / "dataset" / "enriched.jsonl"
    if not src.exists():
        log.warning("skipping year histogram: %s not found", src)
        return None
    doc_year: dict[str, int] = {}
    with src.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            did, yr = r.get("doc_id"), r.get("year")
            if did and isinstance(yr, int) and 1990 <= yr <= 2030 and did not in doc_year:
                doc_year[did] = yr
    if not doc_year:
        log.warning("no valid year metadata in enriched.jsonl")
        return None
    years = np.array(list(doc_year.values()))
    lo, hi = int(years.min()), int(years.max())
    bins = np.arange(lo, hi + 2) - 0.5

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(years, bins=bins, color="#4c78a8", edgecolor="white")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Documents")
    ax.set_title(f"MSQA-Bench source corpus: {len(doc_year):,} documents, "
                 f"{lo}\u2013{hi}")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(lo - 0.5, hi + 0.5)
    fig.tight_layout()
    out = FIG_DIR / "corpus_year_histogram.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main() -> None:
    data = collect()
    log.info("FT models found:   %s", sorted(data["ft"]))
    log.info("Base models found: %s", sorted(data["base"]))
    for fn in (fig_recall_bar, fig_radar, fig_improvement_heatmap):
        out = fn(data)
        log.info("wrote %s", out.relative_to(REPO))
    hist = fig_publication_year_histogram()
    if hist is not None:
        log.info("wrote %s", hist.relative_to(REPO))


if __name__ == "__main__":
    main()
