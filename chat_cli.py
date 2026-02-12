"""
CLI RAG + Gemini Chatbot.

Interactive command-line interface for RAG-powered question answering.
Uses BGE-M3 for retrieval and Google Gemini for answer generation.

Usage:
    python chat_cli.py
"""

import os
from typing import Optional

import google.generativeai as genai
from dotenv import load_dotenv

from src.rag import RAGRetriever, build_prompt, build_single_context
from src.chatlog import ChatLogRepository


def load_env() -> None:
    """Load environment variables from gemini.env or .env file."""
    load_dotenv("gemini.env")
    load_dotenv()


def get_gemini_model() -> "genai.GenerativeModel":
    """Initialize and return Gemini model instance."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Please create a '.env' or 'gemini.env' file "
            "in the project root with: GEMINI_API_KEY=your_real_key"
        )
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    print(f"✅ Đang sử dụng model: gemini-2.5-flash")
    return model


def main() -> None:
    """Main entry point for the CLI chatbot."""
    load_env()
    retriever = RAGRetriever(use_hybrid=True)
    model = get_gemini_model()
    
    try:
        chatlog_repo = ChatLogRepository()
        print("✅ MongoDB connection OK")
    except Exception as exc:
        print(f"❌ MongoDB error: {exc}")
        print("❌ Chat logging sẽ KO được lưu vào database")
        chatlog_repo = None

    print("RAG + Gemini chatbot tren du lieu Facebook group.")
    print("Nhap cau hoi (hoac 'exit' de thoat).")
    print()

    # Ngưỡng điểm tối thiểu để coi là có dữ liệu phù hợp
    MIN_SCORE_THRESHOLD = 0.3
    
    conversation_id: Optional[str] = None
    is_first_message = True

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            if conversation_id:
                print(f"\n✅ Cuộc trò chuyện đã được lưu: {conversation_id}")
            break

        # 🔵 Tạo/thêm vào conversation
        try:
            if chatlog_repo:
                if is_first_message:
                    # Tạo conversation mới với câu hỏi đầu tiên
                    chat_log = chatlog_repo.create_conversation(question)
                    conversation_id = chat_log.id
                    print(f"💾 Conversation ID: {conversation_id}")
                    is_first_message = False
                else:
                    # Thêm user message vào conversation hiện tại
                    if conversation_id:
                        chatlog_repo.add_user_message(conversation_id, question)
        except Exception as exc:
            print(f"❌ Lỗi khi tạo conversation: {exc}")
            chatlog_repo = None
            continue

        # Lấy top 5 documents để có thể hiển thị danh sách liên quan
        docs = retriever.retrieve(question, top_k=5)
        
        if not docs:
            print("\n--- Bot (Gemini) ---")
            answer = "Hiện chưa có dữ liệu để trả lời câu hỏi này."
            print(answer)
            # Lưu assistant response (không có sources)
            if chatlog_repo and conversation_id:
                try:
                    chatlog_repo.add_assistant_message(conversation_id, answer, [])
                except Exception as exc:
                    print(f"⚠️ Không thể lưu response: {exc}")
            continue

        # Kiểm tra điểm của document có độ tương đồng cao nhất
        top_doc = docs[0]
        top_score = top_doc.get("score", 0.0)

        print("\n--- Bot (Gemini) ---")
        
        answer = ""
        sources_for_log = []
        
        # Nếu điểm quá thấp, không có dữ liệu phù hợp
        if top_score < MIN_SCORE_THRESHOLD:
            answer = "Hiện chưa có dữ liệu để trả lời câu hỏi này."
        else:
            # Xác định post để dùng (có thể top_doc là post hoặc comment)
            post_doc = None
            post_id = None
            
            # Kiểm tra xem top_doc có phải là post không
            doc_id = top_doc.get("_id", "")
            if isinstance(doc_id, str) and doc_id.startswith("post::"):
                post_doc = top_doc
                post_id = top_doc.get("source", {}).get("post_id")
            else:
                # Nếu top_doc là comment, lấy post_id và tìm post
                post_id = top_doc.get("source", {}).get("post_id")
                if post_id:
                    # Tìm post trong docs trước
                    for d in docs:
                        d_id = d.get("_id", "")
                        if isinstance(d_id, str) and d_id.startswith("post::"):
                            d_post_id = d.get("source", {}).get("post_id")
                            if d_post_id == post_id:
                                post_doc = d
                                break
                    
                    # Nếu không tìm thấy trong docs, query từ database
                    if not post_doc:
                        post_doc = retriever.get_post_by_id(post_id)
            
            # Nếu tìm được post, dùng post và comments của nó
            if post_doc and post_id:
                context = build_single_context(post_doc, retriever)
            else:
                # Nếu không tìm được post, chỉ dùng document đó (không có comments)
                meta = top_doc.get("source", {})
                link = meta.get("permalink_url") or ""
                context = f"text: {top_doc['text']}\nsource: {link}"
            
            prompt = build_prompt(question, context)

            try:
                resp = model.generate_content(prompt)
                answer = getattr(resp, "text", "").strip() or "[Khong nhan duoc text tu Gemini]"
                # Kiểm tra nếu Gemini trả lời không có dữ liệu
                if not answer or answer.lower() in ["không rõ", "không có", "không tìm thấy"]:
                    answer = "Hiện chưa có dữ liệu để trả lời câu hỏi này."
            except Exception as exc:
                answer = f"Loi khi goi Gemini: {exc}"
            
            # Chuẩn bị sources để lưu
            for d in docs:
                src_meta = d.get("source", {})
                link = src_meta.get("permalink_url") or ""
                sources_for_log.append({
                    "link": link,
                    "text": d.get("text", ""),
                    "score": float(d.get("score", 0.0)),
                    "dense_score": float(d.get("dense_score", 0.0)),
                })
        
        print(answer)
        
        # 💾 Lưu assistant response vào chatlog
        if chatlog_repo and conversation_id:
            try:
                chatlog_repo.add_assistant_message(conversation_id, answer, sources_for_log)
                print("✅ Đã lưu response vào chatlog")
            except Exception as exc:
                print(f"⚠️ Không thể lưu response: {exc}")

        # Hiển thị link bài viết đã dùng để trả lời (top 1) và các bài viết liên quan
        if top_score >= MIN_SCORE_THRESHOLD:
            # Xác định post để hiển thị link (có thể top_doc là post hoặc comment)
            post_doc = None
            doc_id = top_doc.get("_id", "")
            if isinstance(doc_id, str) and doc_id.startswith("post::"):
                post_doc = top_doc
            else:
                # Nếu top_doc là comment, tìm post tương ứng
                post_id = top_doc.get("source", {}).get("post_id")
                if post_id:
                    for d in docs:
                        d_id = d.get("_id", "")
                        if isinstance(d_id, str) and d_id.startswith("post::"):
                            if d.get("source", {}).get("post_id") == post_id:
                                post_doc = d
                                break
                    if not post_doc:
                        post_doc = retriever.get_post_by_id(post_id)
            
            # Hiển thị link của bài viết đã dùng để trả lời
            if post_doc:
                top_src = post_doc.get("source", {})
                top_link = top_src.get("permalink_url", "")
                if top_link:
                    print("\n--- Bài viết đã dùng để trả lời ---")
                    print(f"1. {top_link}")
            
            # Hiển thị các bài viết liên quan (từ bài 2 trở đi)
            if len(docs) > 1:
                print("\n--- Các bài viết liên quan ---")
                for i, d in enumerate(docs[1:], start=2):  # Bắt đầu từ bài thứ 2
                    src = d.get("source", {})
                    link = src.get("permalink_url", "")
                    if link:
                        print(f"{i}. {link}")
                    else:
                        print(f"{i}. [Không có link]")


if __name__ == "__main__":
    main()