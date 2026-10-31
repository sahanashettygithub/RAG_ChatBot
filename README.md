# RAG Document Chatbot

This project lets a user upload a PDF, TXT, or MD document and ask questions about it using a retrieval-augmented generation (RAG) workflow.

## Features

- Upload a document from the browser
- Extract content from PDF or text files
- Split documents into chunks
- Generate embeddings with Google Gemini
- Retrieve relevant context using similarity search
- Answer questions with a Groq LLM

## Setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and add your API keys.

   ```bash
   copy .env.example .env
   ```

3. Start the app:

   ```bash
   streamlit run app/streamlit_app.py
   ```

4. Upload a document in the sidebar and ask questions.

## Files

- `app/rag_chatbot.py` - core document loading and retrieval logic
- `app/streamlit_app.py` - user interface
- `data/sample.txt` - demo document
- `requirements.txt` - dependencies
