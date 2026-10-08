# Evidence — Day 22: LangSmith + Prompt Versioning

**Học viên:** Nguyễn Vũ Huy · 2A202602662
**Môi trường:** OpenAI `gpt-4o-mini` (temperature 0) + `text-embedding-3-small` · FAISS (chunk 500 / overlap 50, k = 3) · Python 3.13 · ragas 0.4.3 · guardrails-ai 0.11.0 · langsmith 0.14.4
**LangSmith project:** `day22-nguyenvuhuy`, có 100 traces (50 `rag-query` + 50 `ab-rag-query`)

## Link LangSmith

| | Link |
|---|---|
| Project (cần đăng nhập) | https://smith.langchain.com/o/51ceb9a5-fe02-4357-9679-4e83b16065c6/projects/p/6dadbdb7-186e-4a8a-90cd-4580f5f7ced6 |
| Trace công khai — Bước 1 `rag-query` (retriever → prompt → LLM → parser) | https://smith.langchain.com/public/bb91fd3e-faa7-45a9-9794-b8f0b3b7cb84/r |
| Trace công khai — Bước 2 `ab-rag-query` (metadata `prompt_version=v2`, `request_id=req-0048`) | https://smith.langchain.com/public/92b555b3-6ca2-4bac-beb2-fc951a006b34/r |

LangSmith hiện chỉ cho chia sẻ công khai **từng trace**, không có chế độ public cho cả project. Vì vậy repo cung cấp link project gốc kèm 2 trace mẫu công khai, mỗi bước 1 trace.

## Danh sách tệp

| Tệp | Nội dung | Tiêu chí |
|---|---|---|
| `01_langsmith_traces.png` | Project lọc theo `name:"rag-query"`, hiển thị **Stats · 50 traces** | 1.3 |
| `01_langsmith_trace_detail.png` | *(bổ sung)* Chi tiết 1 trace: `rag-query` → `VectorStoreRetriever` (3 documents) → `ChatOpenAI` → parser | 1.4 |
| `01_rag_pipeline_log.txt` | *(bổ sung)* Log chạy Bước 1: 50/50 câu hỏi | 1.3 |
| `02_prompt_hub.png` | Prompt Hub có `nguyen-vu-huy-rag-prompt-v1` và `nguyen-vu-huy-rag-prompt-v2` | 2.2 |
| `02_ab_routing_log.txt` | Push → pull cả 2 prompt từ Hub, nhãn `[prompt-v1]`/`[prompt-v2]` cho từng `request_id`, chia V1 = 19 / V2 = 31 | 2.3–2.5 |
| `03_ragas_scores.png` | Biểu đồ + bảng so sánh V1 vs V2 cho 4 metrics (do `03_ragas_evaluation.py` tạo ra) | 3.3 |
| `03_ragas_report.json` | Bản sao `data/ragas_report.json`: điểm V1/V2, thống kê độ dài câu trả lời, điểm từng câu | 3.5 |
| `03_ragas_run_log.txt` | *(bổ sung)* Log đầy đủ của lần chạy RAGAS | 3.1 |
| `04_pii_demo_log.txt` | 8 test case PII: email, phone (2 dạng), SSN, thẻ tín dụng (2 dạng), nhiều PII, câu sạch | 4.2–4.4 |
| `04_json_demo_log.txt` | 6 test case JSON: hợp lệ, fences, nháy đơn, dấu phẩy thừa, lỗi kết hợp, không phải JSON | 4.6–4.8 |

Tất cả ảnh đã làm mờ email tài khoản. Dữ liệu PII trong test case đều là dữ liệu giả.

## Kết quả RAGAS (50 cặp QA × 2 prompt)

| Metric | V1 (ngắn gọn) | V2 (có cấu trúc) | Tốt hơn |
|---|---:|---:|---|
| faithfulness | **0.9648** | 0.9397 | V1 |
| answer_relevancy | **0.9110** | 0.8889 | V1 |
| context_recall | 1.0000 | 1.0000 | hòa |
| context_precision | 0.9417 | 0.9417 | hòa\* |
| Độ dài câu trả lời trung bình | 41.1 từ | 75.6 từ | |

Faithfulness ≥ 0.9 ở **cả 2 phiên bản** (mục tiêu ≥ 0.8).

\* Script in "V2" ở dòng context_precision vì so sánh bằng `>`. Thực tế hai giá trị chỉ lệch nhau khoảng 1e-12 (`0.94166666659408` so với `0.94166666659508`), tức là sai số làm tròn số thực, nên coi là hòa.

## Phân tích: vì sao V1 có điểm cao hơn V2

**1. Faithfulness (0.965 so với 0.940): câu trả lời dài hơn thì có nhiều claim dễ vượt ra ngoài context hơn.**
Faithfulness = số claim được context hỗ trợ / tổng số claim trong câu trả lời.
- V2 yêu cầu "3-5 câu, định nghĩa rồi giải thích chi tiết", nên câu trả lời dài gần gấp đôi V1 (75.6 so với 41.1 từ).
- Mỗi câu thêm vào là thêm claim cần kiểm chứng. Khi model diễn giải hoặc khái quát hóa, ví dụ nêu thêm hệ quả, lợi ích hay so sánh, một vài claim không có nguyên văn trong 3 chunk được retrieve và bị chấm "không suy ra được".
- Dữ liệu từng câu cho thấy V2 thấp hơn V1 ở 16/50 câu, rõ nhất ở *pooling layers* (0.67), *hybrid search* (0.75), *PII* (0.75), *chunking strategy* (0.82). V1 chỉ thấp hơn V2 ở 5 câu.
- Ngoại lệ: V1 bị chấm 0.0 ở câu *"three main types of machine learning"*, dù đáp án có gần như nguyên văn trong knowledge base. Đây là điểm bất thường duy nhất của V1. Report không lưu câu trả lời nên không kiểm chứng được nguyên nhân; có thể do LLM-judge tách claim kém trên một câu trả lời rất ngắn. Bỏ câu này thì faithfulness V1 là 0.9845, nên khoảng cách V1–V2 thực ra còn lớn hơn.

**2. Answer relevancy (0.911 so với 0.889): câu trả lời đi thẳng vào câu hỏi thì được điểm cao hơn.**
RAGAS sinh ngược câu hỏi từ câu trả lời rồi đo độ tương đồng embedding với câu hỏi gốc.
- Câu trả lời ngắn và trực tiếp của V1 sinh ra câu hỏi gần với câu hỏi gốc.
- V2 thêm chi tiết phụ nên câu hỏi sinh ngược bị "loãng". Ví dụ *chunking strategy*: V1 đạt 1.00, V2 đạt 0.68. *Transformer architecture*: V1 0.96, V2 0.79.
- Ở một số ít câu V2 lại cao hơn, ví dụ *context length of GPT-4*: V2 1.00, V1 0.73. Report không lưu câu trả lời nên chưa giải thích được các trường hợp này; có thể là dao động của LLM-judge.

**3. Context recall / precision bằng nhau: hai metric này đo retriever, không đo prompt.**
- Cả hai phiên bản dùng cùng FAISS index, k = 3 và cùng câu hỏi, nên retrieved contexts giống hệt nhau.
- context_recall = 1.0: 3 chunk luôn chứa đủ thông tin của đáp án chuẩn, vì knowledge base có các câu gần như nguyên văn đáp án.
- Chênh lệch nhỏ của context_precision ở 2 câu (*LangSmith*, *PII*) đến từ việc LLM-judge chấm không tất định, không phải do prompt.

**Kết luận.** Với bộ QA dạng định nghĩa ngắn này, **V1 (ngắn gọn) là lựa chọn tốt hơn** cho production: grounded hơn, sát câu hỏi hơn, và câu trả lời ngắn hơn khoảng 46% số từ (tức ít token output hơn). V2 vẫn đạt faithfulness 0.94 nhờ ràng buộc "mọi câu phải được context hỗ trợ". V2 hợp với câu hỏi cần giải thích nhiều bước, nhưng nên thêm giới hạn kiểu "chỉ diễn đạt lại thông tin có trong context" để giảm claim suy diễn.

## Ghi chú

- Bước 3 chạy với `LANGCHAIN_TRACING_V2=false`, để hàng nghìn lời gọi LLM-judge của RAGAS không lẫn vào project LangSmith dùng làm evidence cho Bước 1–2. Prompt V1/V2 trong `03_ragas_evaluation.py` giống nguyên văn prompt đã push lên Hub ở Bước 2.
- Routing tất định: `int(md5(request_id).hexdigest(), 16) % 2`. MD5 không bị salt theo process như `hash()` của Python, nên `req-0000` luôn đi tới V2, `req-0002` luôn đi tới V1, ở mọi lần chạy.
- Guardrails 0.11: chỉ `FailResult(fix_value=...)` mới thay được output khi dùng `OnFailAction.FIX`. Output đã fix vẫn có `validation_passed=True`, nên log JSON in thêm cột `Status` (Valid / Repaired / Fallback) để phân biệt.
