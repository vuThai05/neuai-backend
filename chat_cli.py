import os

from google import genai
from dotenv import load_dotenv

from src.rag import RAGRetriever, build_context, build_prompt


def load_env() -> None:
    """Load environment variables from gemini.env or .env file."""
    load_dotenv("gemini.env")
    load_dotenv()


def get_gemini_client() -> genai.Client:
    """Initialize and return Gemini client instance."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not found. Please create a '.env' or 'gemini.env' file "
        )
    return genai.Client(api_key=api_key)


def main() -> None:
    """Main entry point for the CLI chatbot."""
    load_env()
    retriever = RAGRetriever(use_hybrid=True)
    client = get_gemini_client()

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
            resp = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=prompt
            )
            answer = resp.text.strip() if resp.text else "[Khong nhan duoc text tu Gemini]"
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