# 📄 Document Q&A Assistant

A RAG (Retrieval-Augmented Generation) chatbot that lets you upload a document and ask questions about it. Answers come from the content of your file, not from the model's general knowledge.

**🔗 Live Demo:** [https://ragchatbot-6thkh8qyv6fhej3vwzrfwm.streamlit.app/](https://ragchatbot-6thkh8qyv6fhej3vwzrfwm.streamlit.app/)

## What It Does

Upload a PDF, TXT, or MD file, then ask questions in a chat interface. For example, upload a resume and ask "What is the person's name?" or "What are their skills?"

## How It Works

1. The uploaded document is read and split into small chunks.
2. Each chunk is converted into an embedding and stored in a vector database.
3. When you ask a question, the most relevant chunks are retrieved.
4. The question and retrieved chunks are sent to an LLM, which writes the answer.

## Tech Stack

Python, Streamlit, LangChain, ChromaDB, and an LLM API (Gemini / Groq / OpenAI).

## Run Locally

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
pip install -r requirements.txt
streamlit run app.py
```

Add your API key in a `.env` file before running:

```env
GOOGLE_API_KEY=your_api_key
```

<img width="1632" height="927" alt="image" src="https://github.com/user-attachments/assets/8be99ce8-d4bc-4966-91b5-6c868b94ef05" />
