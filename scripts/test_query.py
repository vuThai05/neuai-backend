#!/usr/bin/env python3
"""Quick test script for RAG retrieval functionality."""

import os

from google import genai
from dotenv import load_dotenv

from src.rag import RAGRetriever, build_context, build_prompt

load_dotenv("gemini.env")
load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY not found. Will only display RAG retrieval results.")
    client = None
else:
    client = genai.Client(api_key=api_key)

print("Loading RAG retriever...")
retriever = RAGRetriever(use_hybrid=True)

query = "thầy Phùng Ngọc Tùng dạy cái gì"
print(f"\n{'='*60}")
print(f"TEST QUERY: {query}")
print(f"{'='*60}\n")

print("Searching knowledge base...")
docs = retriever.retrieve(query, top_k=5)

print(f"\nFound {len(docs)} results:\n")
for i, d in enumerate(docs, start=1):
    dense_score = d.get("dense_score", d.get("score"))
    final_score = d.get("score", dense_score)
    print(f"[DOC {i}]")
    print(f"  Final Score: {final_score:.3f}")
    if "dense_score" in d:
        print(f"  Dense Score: {dense_score:.3f}")
    print(f"  Text: {d['text'][:200]}...")
    source = d.get("source", {})
    if source.get("permalink_url"):
        print(f"  Link: {source['permalink_url']}")
    print()

if client:
    print(f"\n{'='*60}")
    print("GEMINI ANSWER:")
    print(f"{'='*60}\n")
    
    context = build_context(docs)
    prompt = build_prompt(query, context)
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        answer = resp.text.strip() if resp.text else "[No response from Gemini]"
        print(answer)
    except Exception as e:
        print(f"Error calling Gemini: {e}")
else:
    print("\n(No Gemini API key available, showing RAG results only)")

