## Phase 1 - UX & API polish (quick wins)
- [ ] UX: Bổ sung meta social (loại post/comment, thời gian, like/comment/share, group, người đăng) vào context trả về (từ `source` hoặc các field Mongo hiện có) để frontend hiển thị nguồn rõ ràng hơn.
- [ ] UX: Cải thiện hành vi khi không có tài liệu phù hợp (context rỗng) – backend trả về message rõ ràng hoặc flag để frontend hiển thị “không tìm thấy bài viết phù hợp, hãy thử từ khóa khác”.
- [ ] UX: Chuẩn hoá mức độ tin cậy (high/medium/low) dựa trên `score`/`dense_score` và trả kèm theo từng source cho frontend.
- [ ] API: Cho phép client gửi `top_k` (giới hạn tối đa, ví dụ 10) trong request để linh hoạt hơn theo UI/UX của frontend.
- [ ] API: Chuẩn hoá format lỗi (400/503/500) với mã lỗi nội bộ (`code`, `message`) để frontend dễ xử lý (ví dụ: `MONGO_CONFIG_ERROR`, `NO_EMBEDDINGS`, `GEMINI_ERROR`).

## Phase 2 - RAG quality & retrieval tuning
- [ ] RAG: Expose tham số `min_score`, `dense_weight`, `sparse_weight` qua config/env để dễ tuning mà không phải sửa code.
- [ ] RAG: Log ra top_k docs (id rút gọn, score, dense_score, vài ký tự đầu của text) trong môi trường dev để quan sát chất lượng retrieval theo từng cấu hình.
- [ ] RAG: Thiết kế một vài câu hỏi mẫu (FAQ sinh viên, các chủ đề nóng) để đánh giá thủ công chất lượng retrieval và từ đó điều chỉnh `dense_weight`/`sparse_weight`.
- [ ] RAG: Cân nhắc cải thiện phần sparse search (hiện tại chỉ nhân trọng số token) – thử nghiệm chuẩn hoá theo độ dài văn bản hoặc tích hợp BM25 chuẩn nếu cần.
 - [ ] RAG/Reasoning: Thiết kế lại prompt và luồng gọi LLM để LLM phân tích ý định câu hỏi (muốn tìm post tổng quan, hay chi tiết comment/thảo luận) rồi giải thích reasoning và ưu tiên lựa chọn các bài post/comment phù hợp trong câu trả lời.

## Phase 3 - Hiệu suất & khả năng mở rộng
- [ ] Perf: Đo và log riêng thời gian cho từng bước trong `/chat` (retriever, build_context, LLM call, mapping sources) để xác định bottleneck.
- [ ] Perf: Đánh giá chi phí RAM khi load toàn bộ embeddings vào bộ nhớ; nếu số lượng bài viết lớn, cân nhắc:
  - Dùng MongoDB Atlas vector search cho dense part, hoặc
  - Phân shard knowledge base theo group/thời gian và chỉ load subset cần thiết.
- [ ] Perf: Thêm tuỳ chọn bật/tắt hybrid search (dense + sparse) qua config để có thể so sánh hiệu năng/độ chính xác giữa hai chế độ.

## Phase 4 - Observability & robustness
- [ ] Obs: Thêm endpoint `/health` để kiểm tra nhanh trạng thái Mongo, retriever (embeddings đã load chưa), GEMINI_API_KEY, phục vụ monitoring/deployment.
- [ ] Obs: Tăng cường logging có cấu trúc (JSON log) cho `/chat` với các field: `question_preview`, `top_k`, `used_hybrid`, `latency_ms`, `model`, `error_code` (nếu có).
- [ ] Obs: Thiết lập cơ chế fallback cho LLM (ví dụ: nếu Gemini lỗi/timeout, trả về thông báo thân thiện + context rút gọn cho người dùng thay vì lỗi 500 trống.
