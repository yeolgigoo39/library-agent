"""Ollama 본선 답변용 모델."""

from langchain_ollama import ChatOllama

# 과제에서 지정한 로컬 모델
MODEL = "qwen3:4b-instruct-2507-q4_K_M"

# project_4와 같이 temperature=0, num_ctx=2048
main_model = ChatOllama(model=MODEL, temperature=0.0, num_ctx=2048)
