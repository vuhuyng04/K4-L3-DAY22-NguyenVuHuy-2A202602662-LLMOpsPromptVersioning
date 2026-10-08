"""
Bước 3 — RAGAS Evaluation
===========================
NHIỆM VỤ:
  1. Chạy 50 QA pairs qua CẢ 2 prompt version, lưu answers + contexts
  2. Tạo EvaluationDataset với các SingleTurnSample object
  3. Đánh giá với 4 RAGAS metrics: faithfulness, answer_relevancy,
     context_recall, context_precision
  4. In bảng so sánh V1 vs V2
  5. Lưu kết quả vào data/ragas_report.json (+ bản sao evidence/03_ragas_report.json
     và biểu đồ evidence/03_ragas_scores.png)

DELIVERABLE: faithfulness ≥ 0.8 cho ít nhất 1 prompt version
             + file data/ragas_report.json được tạo ra

⏰ LƯU Ý: Bước này mất ~15-30 phút. Hãy bắt đầu sớm!
"""
import sys
import json
import math
import warnings
from datetime import datetime, timezone
warnings.filterwarnings("ignore")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

try:
    from ragas import RunConfig
except ImportError:  # ragas cũ
    from ragas.run_config import RunConfig

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import QA_PAIRS


METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
RETRIEVER_K  = 3
ROOT_DIR     = Path(__file__).parent.parent


# ── 1. Prompt Templates (copy nguyên văn từ Bước 2) ───────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant. Answer the user's question using ONLY the context below. "
    "Keep the answer short and direct (2-4 sentences). "
    "If the context does not contain the answer, say that you don't know. "
    "Always answer in English.\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI analyst. Read the context carefully, identify the facts that are "
    "relevant to the question, and then write a clear, well-organized answer of 3-5 sentences: "
    "start with a one-sentence definition or direct answer, then explain the key details. "
    "Every statement must be supported by the context; do not add outside knowledge, "
    "examples, or speculation. If the context is insufficient, say so explicitly. "
    "Always answer in English.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}


# ── 2. Setup Vectorstore ───────────────────────────────────────────────────
def setup_vectorstore():
    """Tái sử dụng — tạo FAISS vectorstore từ knowledge base (chunk 500/50)."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 3. Chạy RAG và thu thập kết quả ───────────────────────────────────────
def run_rag(retriever, llm, prompt, question: str) -> dict:
    """
    Chạy RAG chain cho 1 câu hỏi.

    ⚠️ QUAN TRỌNG: trả về contexts là LIST of strings, KHÔNG phải string đã ghép!
    RAGAS cần từng đoạn riêng để tính context_recall và context_precision.

    Trả về: {"answer": str, "contexts": list[str]}
    """
    docs     = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]   # list[str] cho RAGAS
    ctx_str  = "\n\n".join(contexts)                 # chỉ dùng để điền {context} của prompt

    answer = (prompt | llm | StrOutputParser()).invoke({
        "context":  ctx_str,
        "question": question,
    })

    return {"answer": answer, "contexts": contexts}


def collect_rag_outputs(vectorstore, prompt_version: str) -> list:
    """
    Chạy tất cả 50 QA pairs qua prompt version được chỉ định.
    Trả về: list of dict với keys: question, reference, answer, contexts

    Câu bị lỗi (rate limit, timeout...) vẫn được giữ với answer rỗng để
    đủ 50 sample — điểm của nó sẽ thấp chứ không âm thầm biến mất.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm       = get_llm()
    prompt    = PROMPTS[prompt_version]

    results = []
    print(f"\n🚀 Đang chạy {len(QA_PAIRS)} câu hỏi với prompt {prompt_version} ...")

    for i, qa in enumerate(QA_PAIRS, 1):
        try:
            out = run_rag(retriever, llm, prompt, qa["question"])
        except Exception as e:
            print(f"  ❌ [{i:02d}] Lỗi: {e}")
            out = {"answer": "", "contexts": []}

        results.append({
            "question":  qa["question"],
            "reference": qa["reference"],
            "answer":    out["answer"],
            "contexts":  out["contexts"],
        })
        print(f"  [{i:02d}/{len(QA_PAIRS)}] {qa['question'][:60]}")

    return results


# ── 4. Tạo RAGAS EvaluationDataset ────────────────────────────────────────
def build_ragas_dataset(rag_results: list) -> EvaluationDataset:
    """
    Chuyển đổi kết quả RAG thành RAGAS EvaluationDataset.

    Mỗi SingleTurnSample cần 4 trường:
      user_input         → câu hỏi
      response           → câu trả lời đã tạo
      retrieved_contexts → list[str] các đoạn đã retrieve
      reference          → đáp án chuẩn (ground truth)
    """
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    return EvaluationDataset(samples=samples)


# ── 5. Chạy RAGAS Evaluation ──────────────────────────────────────────────
def _clean(v):
    """Chuyển NaN/None → None để ghi JSON hợp lệ."""
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) else round(v, 4)


def run_ragas_eval(rag_results: list, version: str) -> tuple:
    """
    Đánh giá kết quả RAG với 4 RAGAS metrics.

    Trả về: (scores, per_sample)
      scores     : {metric_name: mean_score} (bỏ qua sample NaN)
      per_sample : list điểm từng câu hỏi — dùng để phân tích V1 vs V2

    Lưu ý: evaluate() thực hiện rất nhiều lần gọi LLM → mất 5-10 phút / version.
    """
    print(f"\n📐 Đang đánh giá RAGAS cho prompt {version} ... (vui lòng chờ ~5-10 phút)")

    dataset = build_ragas_dataset(rag_results)

    # LLM và Embeddings riêng để RAGAS dùng làm evaluator
    llm_eval = get_llm(temperature=0)
    emb_eval = get_embeddings()

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=llm_eval,
        embeddings=emb_eval,
        run_config=RunConfig(max_workers=8, timeout=180, max_retries=10),
    )

    # result[metric] trả về list điểm theo từng sample → lấy trung bình, bỏ NaN
    scores = {}
    for key in METRIC_NAMES:
        raw = [v for v in result[key] if v is not None]
        scores[key] = float(np.nanmean(raw)) if raw else float("nan")

    per_sample = [
        {"question": r["question"], **{k: _clean(result[k][i]) for k in METRIC_NAMES}}
        for i, r in enumerate(rag_results)
    ]

    print(f"\n📊 Kết quả RAGAS — Prompt {version.upper()}:")
    for k, v in scores.items():
        star = " ⭐" if k == "faithfulness" and v >= 0.8 else ""
        print(f"  {k:30s}: {v:.4f}{star}")

    return scores, per_sample


# ── 6. Báo cáo & biểu đồ ──────────────────────────────────────────────────
def answer_stats(rag_results: list) -> dict:
    """Thống kê độ dài câu trả lời — giúp giải thích khác biệt giữa V1 và V2."""
    words = [len(r["answer"].split()) for r in rag_results]
    return {
        "avg_answer_words": round(float(np.mean(words)), 1),
        "empty_answers":    sum(1 for r in rag_results if not r["answer"].strip()),
    }


def save_scores_chart(v1_scores: dict, v2_scores: dict, path: Path):
    """Vẽ bảng + biểu đồ cột V1 vs V2 cho 4 metrics (bỏ qua nếu chưa cài matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("ℹ️  Chưa cài matplotlib — bỏ qua biểu đồ.")
        return

    v1 = [v1_scores[m] for m in METRIC_NAMES]
    v2 = [v2_scores[m] for m in METRIC_NAMES]
    x  = np.arange(len(METRIC_NAMES))

    fig, (ax, ax_tab) = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [3, 1.3]})
    b1 = ax.bar(x - 0.2, v1, 0.4, label="V1 (concise)",    color="#4C78A8")
    b2 = ax.bar(x + 0.2, v2, 0.4, label="V2 (structured)", color="#F58518")
    ax.bar_label(b1, fmt="%.3f", fontsize=9)
    ax.bar_label(b2, fmt="%.3f", fontsize=9)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="faithfulness target 0.8")
    ax.set_xticks(x, METRIC_NAMES)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("score")
    ax.set_title(f"RAGAS — Prompt V1 vs V2 ({len(QA_PAIRS)} QA pairs, {config.PROVIDER})")
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)

    ax_tab.axis("off")
    rows = [[m, f"{a:.4f}", f"{b:.4f}", "V1" if a > b else ("V2" if b > a else "tie")]
            for m, a, b in zip(METRIC_NAMES, v1, v2)]
    tab = ax_tab.table(cellText=rows, colLabels=["Metric", "V1", "V2", "Winner"],
                       loc="center", cellLoc="center")
    tab.scale(1, 1.4)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"🖼️  Đã lưu biểu đồ vào {path}")


# ── 7. Main ────────────────────────────────────────────────────────────────
def main():
    """Thu thập output V1/V2, chấm RAGAS, in bảng so sánh, lưu report + biểu đồ."""
    print("=" * 60)
    print("  Bước 3: RAGAS Evaluation")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore = setup_vectorstore()

    # Thu thập kết quả RAG cho cả V1 và V2
    v1_results = collect_rag_outputs(vectorstore, "v1")
    v2_results = collect_rag_outputs(vectorstore, "v2")

    # Chạy RAGAS evaluation
    v1_scores, v1_samples = run_ragas_eval(v1_results, "v1")
    v2_scores, v2_samples = run_ragas_eval(v2_results, "v2")

    # In bảng so sánh
    print("\n" + "=" * 65)
    print(f"  {'Metric':30s}  {'V1':>8}  {'V2':>8}  Winner")
    print("=" * 65)
    winners = {}
    for metric in METRIC_NAMES:
        s1, s2  = v1_scores[metric], v2_scores[metric]
        winners[metric] = "v1" if s1 > s2 else ("v2" if s2 > s1 else "tie")
        label = {"v1": "← V1", "v2": "← V2", "tie": "= tie"}[winners[metric]]
        print(f"  {metric:30s}  {s1:>8.4f}  {s2:>8.4f}  {label}")

    # Kiểm tra mục tiêu
    best_faith = max(v1_scores["faithfulness"], v2_scores["faithfulness"])
    if best_faith >= 0.8:
        print(f"\n✅ Đạt mục tiêu: faithfulness = {best_faith:.4f} ≥ 0.8")
    else:
        print(f"\n⚠️  Chưa đạt mục tiêu ({best_faith:.4f} < 0.8).")
        print("   Gợi ý: giảm chunk_size, tăng k, hoặc điều chỉnh prompt.")

    # Lưu báo cáo
    report = {
        "prompt_v1_scores": v1_scores,
        "prompt_v2_scores": v2_scores,
        "target_met": best_faith >= 0.8,
        "winner_per_metric": winners,
        "answer_stats": {"v1": answer_stats(v1_results), "v2": answer_stats(v2_results)},
        "config": {
            "provider":         config.PROVIDER,
            "llm_model":        config.OPENAI_MODEL if config.PROVIDER == "openai" else config.PROVIDER,
            "embedding_model":  config.OPENAI_EMBEDDING_MODEL if config.PROVIDER == "openai" else config.PROVIDER,
            "retriever_k":      RETRIEVER_K,
            "chunk_size":       500,
            "chunk_overlap":    50,
            "n_samples":        len(QA_PAIRS),
            "generated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "per_sample": {"v1": v1_samples, "v2": v2_samples},
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)

    report_path = ROOT_DIR / "data" / "ragas_report.json"
    report_path.write_text(text, encoding="utf-8")
    print(f"💾 Đã lưu báo cáo vào {report_path}")

    # data/ragas_report.json bị gitignore → ghi thêm bản sao vào evidence/
    evidence_dir = ROOT_DIR / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    (evidence_dir / "03_ragas_report.json").write_text(text, encoding="utf-8")
    print(f"💾 Đã sao chép báo cáo vào {evidence_dir / '03_ragas_report.json'}")

    save_scores_chart(v1_scores, v2_scores, evidence_dir / "03_ragas_scores.png")


# ── Phân tích kết quả (lần chạy nộp bài, gpt-4o-mini, 50 QA) ─────────────
#   faithfulness      V1 0.9648 > V2 0.9397  — V2 dài gấp ~1.8 lần (75.6 vs 41.1 từ)
#                     → nhiều claim hơn, một số là diễn giải không có nguyên văn trong context.
#   answer_relevancy  V1 0.9110 > V2 0.8889  — câu trả lời ngắn, trực tiếp sinh ngược ra câu
#                     hỏi sát câu gốc hơn; chi tiết phụ của V2 làm "loãng" câu hỏi sinh ngược.
#   context_recall / context_precision bằng nhau (1.0 / 0.9417) — 2 metric này đo retriever,
#                     mà V1/V2 dùng chung FAISS + k=3; chênh 1e-12 chỉ là sai số số thực.
#   → V1 tốt hơn cho bộ câu hỏi định nghĩa ngắn. Chi tiết: evidence/README.md

if __name__ == "__main__":
    main()
