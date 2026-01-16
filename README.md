# RAG_chatbot

A professional Retrieval-Augmented Generation (RAG) chatbot system using BGE-M3 embeddings and Google Gemini for question answering on Facebook group data.

## 🚀 Features

- **Hybrid Search**: Combines dense (semantic) and sparse (keyword-based) embeddings for superior retrieval accuracy
- **BGE-M3 Model**: State-of-the-art multilingual embedding model from BAAI
- **Gemini Integration**: Powered by Google's Gemini for natural language generation
- **MongoDB Backend**: Efficient document storage and retrieval
- **CLI Interface**: Clean command-line interface for interactive chat

## 📋 Table of Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Workflow](#workflow)

## 🏗️ Architecture

The system follows a RAG (Retrieval-Augmented Generation) pipeline:

1. **Data Indexing**: Posts and comments from MongoDB are normalized into a knowledge base
2. **Embedding Generation**: BGE-M3 model generates both dense and sparse embeddings
3. **Retrieval**: Hybrid search combines semantic and keyword matching
4. **Generation**: Gemini LLM generates answers based on retrieved context

### Key Components

- **RAGRetriever**: Handles document retrieval using hybrid search
- **BGE-M3 Embeddings**: 1024-dimensional dense vectors + sparse token vectors
- **Gemini LLM**: Generates natural language responses

## 📦 Installation

### Prerequisites

- Python 3.8+
- MongoDB (local or cloud instance)
- Google Gemini API key

### Step 1: Clone and Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd chatbot

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

Create a `gemini.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Alternatively, you can use a `.env` file with the same format.

### Step 3: Configure MongoDB

Update `src/utils/config.py` with your MongoDB connection string:

```python
MONGO_URI = "your_mongodb_connection_string"
MONGO_DB_NAME = "your_database_name"
```

## ⚙️ Configuration

The system expects MongoDB collections:
- `posts`: Original post data
- `comments`: Comment data
- `knowledge_base`: Normalized documents for RAG (created by indexing script)

## 🎯 Usage

### Initial Setup (First Time Only)

1. **Index MongoDB data** into knowledge base:
   ```bash
   python scripts/index_mongo.py
   ```

2. **Generate embeddings** for all documents:
   ```bash
   python scripts/embed_bge_m3.py
   ```

### Running the Chatbot

Start the interactive CLI chatbot:

```bash
python chat_cli.py
```

Example interaction:
```
You: thầy Phùng Ngọc Tùng dạy cái gì

--- Bot (Gemini) ---
[Answer from Gemini based on retrieved context]

--- Nguon context ---
[DOC 1] final_score=0.856 (dense=0.712)
  text: [Retrieved document text]
  link: [Source URL]
```

### Testing Retrieval

Test RAG retrieval without full chat:

```bash
python scripts/test_query.py
```

## 📁 Project Structure

```
chatbot/
├── src/
│   ├── __init__.py
│   ├── rag/
│   │   ├── __init__.py
│   │   └── retriever.py      # RAG retriever with hybrid search
│   └── utils/
│       ├── __init__.py
│       └── config.py          # MongoDB configuration
├── scripts/
│   ├── index_mongo.py         # Build knowledge base from MongoDB
│   ├── embed_bge_m3.py        # Generate embeddings
│   └── test_query.py          # Test retrieval
├── chat_cli.py                # Main CLI application
├── requirements.txt           # Python dependencies
├── .gitignore
├── README.md
└── gemini.env                 # Environment variables (not in git)
```

## 📦 Requirements

### Core Dependencies

- `torch>=2.0.0` - PyTorch for deep learning
- `transformers==4.44.2` - Hugging Face transformers
- `FlagEmbedding>=1.2.10` - BGE-M3 embedding model
- `pymongo>=4.10.0` - MongoDB driver
- `python-dotenv>=1.0.1` - Environment variable management
- `numpy>=1.26.0` - Numerical computing
- `google-generativeai>=0.8.0` - Gemini API client

### Model Requirements

- **BGE-M3**: Automatically downloaded on first use (~2GB)
- **GPU**: Optional but recommended for faster embedding generation

## 🔄 Workflow

### Data Pipeline

```
MongoDB (posts/comments)
    ↓
[scripts/index_mongo.py]
    ↓
Knowledge Base (normalized documents)
    ↓
[scripts/embed_bge_m3.py]
    ↓
Knowledge Base + Embeddings
    ↓
[chat_cli.py]
    ↓
User Query → RAG Retrieval → Gemini → Answer
```

### Retrieval Process

1. **Query Encoding**: User query is encoded into dense and sparse vectors
2. **Similarity Calculation**: 
   - Dense: Cosine similarity with document embeddings
   - Sparse: BM25-like token matching
3. **Hybrid Scoring**: Weighted combination (default: 70% dense, 30% sparse)
4. **Top-K Selection**: Returns top 5 most relevant documents
5. **Context Building**: Documents formatted for LLM
6. **Generation**: Gemini generates answer from context

### Score Interpretation

- **Final Score [0, 1]**: Combined hybrid score
  - `> 0.7`: Excellent match
  - `0.5-0.7`: Good match
  - `< 0.4`: May not be relevant

- **Dense Score [-1, 1]**: Cosine similarity
  - `> 0.7`: Strong semantic match
  - `0.3-0.7`: Moderate semantic match

## 🔧 Advanced Configuration

### RAGRetriever Parameters

You can customize the retriever in `chat_cli.py`:

```python
retriever = RAGRetriever(
    collection_name="knowledge_base",  # MongoDB collection
    top_k=5,                             # Number of documents to retrieve
    use_hybrid=True,                     # Enable hybrid search
    dense_weight=0.7,                    # Weight for dense similarity
    sparse_weight=0.3,                   # Weight for sparse similarity
    min_score=0.4,                       # Minimum score threshold (optional)
)
```

### Embedding Generation

Customize batch size and sparse embeddings in `scripts/embed_bge_m3.py`:

```python
embed_knowledge_base(
    batch_size=16,        # Larger batch = faster but more memory
    use_sparse=True,      # Enable sparse embeddings for hybrid search
)
```

## 🐛 Troubleshooting

### "Khong tim thay embedding trong Mongo"

Run the embedding generation script:
```bash
python scripts/embed_bge_m3.py
```

### "Khong tim thay GEMINI_API_KEY"

Ensure `gemini.env` exists with:
```
GEMINI_API_KEY=your_key_here
```

### Slow Retrieval

- Embeddings are cached in RAM on startup
- First query may be slower (model loading)
- Consider reducing `top_k` if needed

## 📝 License

[Specify your license here]

## 👥 Contributing

[Add contribution guidelines if applicable]

## 📧 Contact

[Add contact information if needed]

