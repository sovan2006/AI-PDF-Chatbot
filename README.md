# 🧠 AI Knowledge Assistant (RAG)

> **An intelligent PDF chatbot powered by Retrieval-Augmented Generation (RAG), Hybrid Search, and Large Language Models.** Upload PDF documents, build a semantic knowledge base, and chat with your documents using natural language.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector_DB-orange?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Database-purple?style=for-the-badge)
![Sentence Transformers](https://img.shields.io/badge/SentenceTransformers-Embeddings-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

</p>

---

# 🚀 Live Demo

### 🌐 Web App

> **https://ai-pdf-chatbot-umrwim3hxtxnezsrnzdeae.streamlit.app/**

---

# 📸 Screenshots

## Home Page

<img width="1440" height="900" alt="Screenshot 2026-07-16 at 8 54 02 AM" src="https://github.com/user-attachments/assets/cb71348e-7c49-4a37-a453-9affbabf78ba" />


## Chat Interface

<img width="1440" height="900" alt="Screenshot 2026-07-16 at 8 55 30 AM" src="https://github.com/user-attachments/assets/755b784d-da28-460d-ab18-2cc77cfb9e52" />


---

# 📖 Overview

AI Knowledge Assistant is a production-ready Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions.

Instead of relying only on the LLM's internal knowledge, the system retrieves relevant document chunks using Hybrid Search and feeds them into the LLM to generate accurate, grounded answers.

The application combines:

- Dense Semantic Search
- BM25 Keyword Search
- Reciprocal Rank Fusion (RRF)
- Context-aware Prompt Engineering
- Source Attribution
- PDF Knowledge Base

This significantly reduces hallucinations while improving answer quality.

---

# ✨ Features

## 📄 PDF Knowledge Base

- Upload one or multiple PDFs
- Automatic document parsing
- Metadata extraction
- Persistent vector storage

---

## ✂️ Smart Text Chunking

- Recursive Character Text Splitter
- Configurable chunk size
- Configurable overlap
- Preserves context

---

## 🧠 Semantic Embeddings

Generate high-quality vector embeddings using Sentence Transformers.

Supports:

- BAAI
- HuggingFace
- MiniLM
- MPNet
- BGE Models

---

## 🔍 Hybrid Retrieval

Instead of relying only on vector similarity, this project combines:

✔ Dense Retrieval

✔ BM25 Retrieval

✔ Reciprocal Rank Fusion (RRF)

This dramatically improves retrieval accuracy.

---

## 🤖 LLM Powered Answers

Uses Large Language Models to answer questions using retrieved context.

Responses include:

- Grounded answers
- Source citations
- Context-aware explanations
- Reduced hallucinations

---

## 📚 Source Attribution

Every answer includes document references.

Example:

```
Transformer replaces RNNs using self-attention.
(Source 2)

Residual connections improve optimization.
(Source 5)
```

---

## 💾 Persistent Knowledge Base

Documents are stored locally inside:

```
vectorstore/
```

No need to re-index after restarting.

---

# 🏗️ System Architecture

```
                  Upload PDF
                       │
                       ▼
             PDF Text Extraction
                       │
                       ▼
                Text Cleaning
                       │
                       ▼
          Recursive Chunk Splitting
                       │
                       ▼
            Generate Embeddings
                       │
                       ▼
         Store in ChromaDB / FAISS
                       │
──────────────────────────────────────────────

               User Question
                       │
                       ▼
          Generate Query Embedding
                       │
                       ▼
        Hybrid Retrieval (Dense + BM25)
                       │
                       ▼
          Reciprocal Rank Fusion
                       │
                       ▼
            Top Relevant Chunks
                       │
                       ▼
            Prompt Construction
                       │
                       ▼
                   LLM
                       │
                       ▼
      Grounded Answer + Sources
```

---

# 🛠️ Tech Stack

## Frontend

- Streamlit

---

## Backend

- Python

---

## AI Framework

- LangChain

---

## Embedding Models

- Sentence Transformers
- HuggingFace Embeddings

---

## Vector Database

- ChromaDB
- FAISS

---

## Search Algorithms

- Dense Search
- BM25
- Hybrid Retrieval
- Reciprocal Rank Fusion

---

## PDF Processing

- PyPDF
- LangChain Document Loaders

---

## LLM

Supports:

- Groq
- OpenAI
- Gemini
- Ollama
- HuggingFace

---

# 📂 Project Structure

```
AI-Knowledge-Assistant/
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── .env
│
├── uploads/
│
├── vectorstore/
│
├── data/
│
├── models/
│
├── utils/
│   ├── loader.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── vectorstore.py
│   ├── prompt.py
│   └── llm.py
│
├── assets/
│   ├── home.png
│   └── chat.png
│
└── notebooks/
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AI-Knowledge-Assistant.git

cd AI-Knowledge-Assistant
```

---

## Create Virtual Environment

Mac/Linux

```bash
python3 -m venv venv

source venv/bin/activate
```

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

Create

```
.env
```

```
GROQ_API_KEY=YOUR_API_KEY

OPENAI_API_KEY=YOUR_API_KEY

GOOGLE_API_KEY=YOUR_API_KEY
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 💡 How It Works

## Step 1

Upload PDF

↓

## Step 2

Extract Text

↓

## Step 3

Clean Text

↓

## Step 4

Split into Chunks

↓

## Step 5

Generate Embeddings

↓

## Step 6

Store in Vector Database

↓

## Step 7

Ask Question

↓

## Step 8

Convert Query into Embedding

↓

## Step 9

Hybrid Retrieval

↓

## Step 10

Top-K Context Retrieved

↓

## Step 11

LLM Generates Answer

↓

## Step 12

Display Answer with Sources

---

# 📊 Performance

| Metric | Value |
|----------|-------|
| Chunking Strategy | Recursive |
| Search Method | Hybrid |
| Retrieval | Dense + BM25 |
| Fusion | RRF |
| Vector Database | ChromaDB |
| Embedding Model | Sentence Transformers |
| UI | Streamlit |
| Source Citation | ✅ |
| Persistent Storage | ✅ |

---

# 🎯 Future Improvements

- Multi-document chat
- Chat history
- OCR support
- Image understanding
- Table extraction
- Voice input
- Speech output
- Agentic RAG
- Re-ranking models
- Graph RAG
- Knowledge Graph
- User authentication
- Cloud deployment
- Docker support
- Kubernetes deployment

---

# 💻 Example Questions

```
What is Transformer?

Explain Attention Mechanism.

Summarize Chapter 4.

Who proposed this research?

What are the main contributions?

Explain the methodology.

Generate interview questions from this PDF.

Summarize the entire paper.

List important formulas.

Compare Encoder and Decoder.
```

---

# 📈 Why This Project?

This project demonstrates real-world AI engineering skills including:

- Retrieval-Augmented Generation (RAG)
- Hybrid Search
- Vector Databases
- Embeddings
- Prompt Engineering
- LLM Integration
- LangChain
- Streamlit
- Information Retrieval
- NLP
- Semantic Search

It is an excellent portfolio project for:

- AI Engineer
- Machine Learning Engineer
- NLP Engineer
- Generative AI Engineer
- LLM Engineer

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository

2. Create a feature branch

```bash
git checkout -b feature-name
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push changes

```bash
git push origin feature-name
```

5. Open a Pull Request

---

# 📜 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

## **Sovan Barik**

**AI & Machine Learning Engineer**

- 🧠 Artificial Intelligence
- 🤖 Machine Learning
- 📚 NLP
- 🔍 Retrieval-Augmented Generation (RAG)
- ⚡ LangChain
- 🚀 Generative AI

---

## ⭐ If you found this project useful, don't forget to Star the repository!
