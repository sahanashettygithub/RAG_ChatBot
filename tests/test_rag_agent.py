from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app import rag_chatbot


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def similarity_search(self, query, k):
        self.queries.append((query, k))
        return self.documents


class FakeChatModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content="Answer from document context.")


def test_rag_agent_retrieves_context_before_answering(monkeypatch):
    vector_store = FakeVectorStore([Document(page_content="The project uses Python.")])
    model = FakeChatModel()
    monkeypatch.setattr(rag_chatbot, "ChatGroq", lambda **kwargs: model)

    agent = rag_chatbot.create_rag_agent(vector_store)
    result = agent("Which language does it use?", [])

    assert vector_store.queries == [("Which language does it use?", 3)]
    assert "The project uses Python." in model.messages[0].content
    assert result == "Answer from document context."


def test_rag_agent_handles_no_matching_context(monkeypatch):
    vector_store = FakeVectorStore([])
    model = FakeChatModel()
    monkeypatch.setattr(rag_chatbot, "ChatGroq", lambda **kwargs: model)

    agent = rag_chatbot.create_rag_agent(vector_store)
    agent("What is in this file?", [])

    assert "No relevant content was found in the uploaded document." in model.messages[0].content
