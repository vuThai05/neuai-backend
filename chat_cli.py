"""
CLI RAG + Gemini Chatbot.

Interactive command-line interface for RAG-powered question answering.
Uses BGE-M3 for retrieval and Google Gemini for answer generation.

Usage:
    python chat_cli.py
"""

import os

import google.generativeai as genai
from dotenv import load_dotenv

from src.rag import RAGRetriever, build_context, build_prompt


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
    return genai.GenerativeModel("gemini-1.5-flash")


def main() -> None:
    """Main entry point for the CLI chatbot."""
    load_env()
    retriever = RAGRetriever(use_hybrid=True)
    model = get_gemini_model()

    print("RAG + Gemini chatbot tren du lieu Facebook group.")
    print("Nhap cau hoi (hoac 'exit' de thoat).")

    while True:
        question = input("\nYou: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        docs = retriever.retrieve(question, top_k=5)
        context = build_context(docs)
        prompt = build_prompt(question, context)

        print("\n--- Bot (Gemini) ---")
        try:
            resp = model.generate_content(prompt)
            answer = getattr(resp, "text", "").strip() or "[Khong nhan duoc text tu Gemini]"
        except Exception as exc:
            answer = f"Loi khi goi Gemini: {exc}"
        print(answer)

        print("\n--- Nguon context ---")
        for i, d in enumerate(docs, start=1):
            src = d.get("source", {})
            dense_score = d.get("dense_score")
            if dense_score is not None:
                print(f"[DOC {i}] final_score={d['score']:.3f} (dense={dense_score:.3f})")
            else:
                print(f"[DOC {i}] score={d['score']:.3f}")
            print(f"  text: {d['text']}")
            print(f"  link: {src.get('permalink_url')}\n")


if __name__ == "__main__":
    main()