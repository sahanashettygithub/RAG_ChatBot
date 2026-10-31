from __future__ import annotations

import hashlib
import os

import streamlit as st

from app.rag_chatbot import (
    build_vector_store,
    create_rag_agent,
    load_environment,
    resolve_document_path,
    save_uploaded_file,
)

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄")

load_environment()
AGENT_RUNTIME_VERSION = 1

st.title("📄 Document Q&A Assistant")
st.caption("Upload a PDF, TXT, or MD file and ask questions about it.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        file_digest = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        document_key = f"{file_path}:{file_digest}"
        st.success(f"Loaded: {uploaded_file.name}")
    else:
        file_path = resolve_document_path("data/sample.txt")
        document_key = str(file_path)
        st.info("Using the sample document for a quick demo.")

    clear_chat = st.button("Clear chat", use_container_width=True)

if clear_chat:
    st.session_state.chat_history = []

if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
    st.warning("Add GOOGLE_API_KEY and GROQ_API_KEY to your .env file before using the app.")
    st.stop()

document_changed = st.session_state.get("active_document_key") != document_key
agent_is_stale = (
    not callable(st.session_state.get("agent"))
    or st.session_state.get("agent_runtime_version") != AGENT_RUNTIME_VERSION
)

if document_changed:
    st.session_state.chat_history = []

if document_changed or agent_is_stale:
    try:
        with st.spinner("Indexing the document..."):
            if (
                document_changed
                or "vector_store" not in st.session_state
                or st.session_state.get("vector_store_document_key") != document_key
            ):
                st.session_state.vector_store = build_vector_store(file_path)
                st.session_state.vector_store_document_key = document_key

            st.session_state.agent = create_rag_agent(st.session_state.vector_store)
            st.session_state.agent_runtime_version = AGENT_RUNTIME_VERSION
            st.session_state.active_document_key = document_key
    except Exception:
        st.error("The document could not be indexed. Check that it is readable and try again.")
        st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Ask a question about the uploaded document")
if prompt:
    question = prompt.strip()
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.agent(
                        question,
                        st.session_state.chat_history[:-1],
                    )
                except Exception as error:
                    answer = "I couldn't complete that response. Please check the API connection and try again."
                    error_detail = str(error).strip().splitlines()[0][:400]
                    for name, value in os.environ.items():
                        if value and len(value) >= 8 and any(
                            marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET")
                        ):
                            error_detail = error_detail.replace(value, "[REDACTED]")
                    if error_detail:
                        st.error(f"Request failed ({type(error).__name__}): {error_detail}")
                    else:
                        st.error(f"Request failed ({type(error).__name__}).")
            st.write(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
