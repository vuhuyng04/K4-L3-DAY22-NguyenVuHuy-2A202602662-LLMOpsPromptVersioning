"""
Bước 1 — RAG Pipeline với LangSmith Tracing
=============================================
NHIỆM VỤ:
  1. Tải knowledge base, chia chunks, index với FAISS
  2. Xây dựng RAG chain: retriever → prompt → LLM → output parser
  3. Trang trí hàm query với @traceable để LangSmith ghi lại mỗi lần gọi
  4. Chạy 50 câu hỏi → tạo ≥ 50 traces trên LangSmith

DELIVERABLE: Mở https://smith.langchain.com → project của bạn → xác nhận ≥ 50 traces.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ⚠️ QUAN TRỌNG: Import config TRƯỚC KHI import bất kỳ thư viện LangChain nào.
# config.py tự động đặt LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, ... vào os.environ
import config

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langsmith import traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Thiết lập Vectorstore ───────────────────────────────────────────────
def setup_vectorstore():
    """
    Tải knowledge base, chia chunks (500 ký tự, overlap 50) và tạo FAISS vectorstore.

    Trả về: FAISS vectorstore đã index toàn bộ chunks.
    """
    embeddings = get_embeddings()
    text       = load_knowledge_base()
    chunks     = split_text(text, chunk_size=500, chunk_overlap=50)
    print(f"📚 Đã chia thành {len(chunks)} chunks")

    vectorstore = build_vectorstore(chunks, embeddings)
    return vectorstore


# ── 2. RAG Prompt Template ─────────────────────────────────────────────────
# System message chứa {context} (các chunk được retrieve), human message chứa {question}.
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Bạn là trợ lý AI hữu ích. Chỉ dùng context sau để trả lời.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 3. Build RAG Chain ─────────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Xây dựng LCEL RAG chain theo cấu trúc pipe:
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()

    Trả về: (chain, retriever)
    """
    llm = get_llm()

    # Retriever trả về k=3 chunk gần nhất theo similarity
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        """Ghép page_content của các docs thành 1 chuỗi context."""
        return "\n\n".join(doc.page_content for doc in docs)

    # Retriever nằm TRONG chain → trace trên LangSmith có run con chứa các docs được retrieve
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


# ── 4. Hàm Query có LangSmith Tracing ─────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Chạy RAG chain với một câu hỏi.
    Decorator @traceable gửi mỗi lần gọi lên LangSmith như một trace riêng
    (input, các run con retriever/prompt/LLM/parser, output, latency).
    """
    return chain.invoke(question)


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    """Build pipeline rồi hỏi lần lượt 50 câu trong SAMPLE_QUESTIONS (mỗi câu = 1 trace)."""
    print("=" * 60)
    print("  Bước 1: LangSmith RAG Pipeline")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    vectorstore      = setup_vectorstore()
    chain, retriever = build_rag_chain(vectorstore)

    # Lỗi ở 1 câu (rate limit, timeout...) không làm dừng cả vòng lặp
    n_ok = 0
    for i, question in enumerate(SAMPLE_QUESTIONS, 1):
        print(f"[{i:02d}/{len(SAMPLE_QUESTIONS)}] Q: {question[:60]}")
        try:
            answer = ask(chain, question)
            n_ok += 1
            print(f"       A: {str(answer)[:100]}\n")
        except Exception as e:
            print(f"       ❌ Lỗi: {e}\n")

    if n_ok == 0:
        raise RuntimeError("Không câu hỏi nào chạy thành công — kiểm tra API key / kết nối.")

    print(f"\n✅ {n_ok}/{len(SAMPLE_QUESTIONS)} traces đã gửi lên LangSmith project '{config.LANGSMITH_PROJECT}'")
    print("   Mở https://smith.langchain.com để xem traces.")


if __name__ == "__main__":
    main()
