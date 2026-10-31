import sys

from app.rag_chatbot import run_console_chat


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_console_chat(file_path)



