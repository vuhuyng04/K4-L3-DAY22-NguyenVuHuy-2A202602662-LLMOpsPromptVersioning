"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "nguyen-vu-huy-rag-prompt-v1"
PROMPT_V2_NAME = "nguyen-vu-huy-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
# Câu hỏi và đáp án chuẩn đều bằng tiếng Anh → prompt yêu cầu trả lời bằng tiếng Anh.
# Cả 2 bản đều giữ {context} và chỉ cho phép dùng thông tin trong context.

# V1 — ngắn gọn, trả lời trực tiếp (2-4 câu)
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

# V2 — chuyên gia phân tích, câu trả lời có tổ chức (3-5 câu)
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


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.

    Lỗi 409 "Nothing to commit" khi chạy lại là bình thường: prompt chưa đổi
    nên Hub không tạo phiên bản (commit) mới.
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1,
                                 description="V1 – concise, direct answer (2-4 sentences)")
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        print(f"⚠️  V1 lỗi: {e}")

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2,
                                 description="V2 – structured expert answer (3-5 sentences)")
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        print(f"⚠️  V2 lỗi: {e}")


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub.
    Fallback về template local nếu Hub không khả dụng (in rõ lý do để dễ phát hiện).

    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}
    for name, local in ((PROMPT_V1_NAME, PROMPT_V1), (PROMPT_V2_NAME, PROMPT_V2)):
        try:
            prompts[name] = client.pull_prompt(name)
            print(f"↓ Đã pull '{name}' từ Hub")
        except Exception as e:
            prompts[name] = local
            print(f"ℹ️  Dùng local fallback cho '{name}' (pull lỗi: {e})")
    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.

    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic) — khác với
    random.choice() hay hash() của Python (bị salt khác nhau mỗi process).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.

    Bước:
      a) Retrieve top-3 docs từ retriever
      b) Ghép page_content thành context string
      c) Chạy (prompt | llm | StrOutputParser()).invoke({"context": ..., "question": ...})
      d) Trả về {"question": ..., "answer": ..., "version": ...}
    """
    docs    = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer  = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    """Tạo FAISS vectorstore từ knowledge base (chunk 500/50 như Bước 1)."""
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunks      = split_text(text)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    """Push → pull prompts, rồi định tuyến 50 câu hỏi qua V1/V2 theo hash của request_id."""
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)

    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count, n_err = 0, 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"

        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        try:
            # langsmith_extra gắn metadata vào trace → lọc được theo version trên LangSmith
            result = ask_ab(
                retriever, llm, prompt, question, version_tag,
                langsmith_extra={"metadata": {"request_id": request_id,
                                              "prompt_version": version_tag,
                                              "prompt_name": version_key}},
            )
        except Exception as e:
            n_err += 1
            print(f"[{i+1:02d}] [prompt-{version_tag}] {request_id} ❌ Lỗi: {e}")
            continue

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {request_id} | {question[:55]}...")
        print(f"      → {result['answer'][:90]}")

    if v1_count + v2_count == 0:
        raise RuntimeError("Không câu hỏi nào chạy thành công — kiểm tra API key / kết nối.")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Lỗi={n_err} | Tổng={len(SAMPLE_QUESTIONS)}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
