import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "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")
    LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "default")
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL = "llama3-8b-8192"
    LLM_TEMPERATURE = 0.7