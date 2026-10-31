from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DEFAULT_DOCUMENT = DATA_DIR / "sample.txt"


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def resolve_document_path(file_path: str | Path | None = None) -> Path:
    if file_path is None:
        return DEFAULT_DOCUMENT.resolve()

    candidate = Path(file_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (ROOT / candidate).resolve()


def save_uploaded_file(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = Path(uploaded_file.name).name
    destination = (UPLOAD_DIR / safe_name).resolve()
    content = uploaded_file.getvalue()
    if not destination.exists() or destination.read_bytes() != content:
        destination.write_bytes(content)
    return destination


def load_document(file_path: str | Path) -> list[Document]:
    resolved = resolve_document_path(file_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Document not found: {resolved}")

    suffix = resolved.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(resolved)).load()
    if suffix in {".txt", ".md"}:
        return TextLoader(str(resolved), encoding="utf-8").load()
    raise ValueError("Unsupported file type. Please upload a PDF, TXT, or MD file.")


def build_vector_store(file_path: str | Path | None = None) -> InMemoryVectorStore:
    documents = load_document(file_path)
    if not documents:
        raise ValueError("No readable content was found in the selected document.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)


def create_rag_agent(vector_store: InMemoryVectorStore):
    llm = ChatGroq(model="openai/gpt-oss-20b")

    def answer_question(question: str, chat_history: list[dict[str, str]]) -> str:
        recent_history = chat_history[-12:]
        retrieval_query = question

        if recent_history:
            history_text = "\n".join(
                f"{message['role']}: {message['content']}" for message in recent_history
            )
            rewrite = llm.invoke(
                [
                    SystemMessage(
                        content=(
                            "Rewrite the latest user question as a standalone search query for the "
                            "uploaded document. Resolve references using the conversation. Return "
                            "only the query, and do not answer it."
                        )
                    ),
                    HumanMessage(
                        content=f"Conversation:\n{history_text}\n\nLatest question: {question}"
                    ),
                ]
            )
            rewritten_query = str(rewrite.content).strip()
            if rewritten_query:
                retrieval_query = rewritten_query

        matches = vector_store.similarity_search(query=retrieval_query, k=3)
        context = "\n\n".join(doc.page_content for doc in matches)
        if not context:
            context = "No relevant content was found in the uploaded document."

        messages = [
            SystemMessage(
                content=(
                    "Answer using the uploaded document as the source of truth. Use conversation "
                    "history to understand references, but do not treat previous assistant answers "
                    "as evidence. If the document does not contain the answer, say so.\n\n"
                    f"Retrieved document context:\n{context}"
                )
            )
        ]
        for message in recent_history:
            if message["role"] == "assistant":
                messages.append(AIMessage(content=message["content"]))
            else:
                messages.append(HumanMessage(content=message["content"]))
        messages.append(HumanMessage(content=question))

        response = llm.invoke(messages)
        return str(response.content).strip()

    return answer_question


def run_console_chat(file_path: str | Path | None = None) -> None:
    load_environment()

    missing_keys = [name for name in ("GOOGLE_API_KEY", "GROQ_API_KEY") if not os.getenv(name)]
    if missing_keys:
        raise RuntimeError(
            "Missing API keys: " + ", ".join(missing_keys) + ". Add them to your .env file and try again."
        )

    vector_store = build_vector_store(file_path)
    agent = create_rag_agent(vector_store)
    chat_history: list[dict[str, str]] = []

    print("Type 'quit' to exit.")
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue

        answer = agent(user_input, chat_history)
        chat_history.extend(
            [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": answer},
            ]
        )
        print(f"AI: {answer}")
