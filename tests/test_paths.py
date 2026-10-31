from pathlib import Path

from app.rag_chatbot import ROOT, resolve_document_path


def test_resolve_document_path_uses_project_root():
    resolved = resolve_document_path("data/sample.txt")
    assert resolved == (ROOT / "data" / "sample.txt").resolve()


def test_resolve_document_path_keeps_absolute_path():
    absolute = Path("/tmp/example.txt")
    assert resolve_document_path(str(absolute)) == absolute.resolve()
