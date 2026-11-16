"""Small synchronous adapter around the original EnergyTool consultant."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_RAG = Path(__file__).resolve().parent / "rag"
if str(_RAG) not in sys.path:
    sys.path.insert(0, str(_RAG))

from api_client import APIError, Budget, Config, PlatformAPI  # noqa: E402
from retrieval import Knowledge, Retriever  # noqa: E402
from solution import answer_one  # noqa: E402


class RAGUnavailableError(RuntimeError):
    """The configured LLM did not produce a usable answer."""


class RAGService:
    def __init__(self, knowledge_path: str | Path | None = None, config: Config | None = None):
        self.config = config or Config.load()
        if config is None:
            # A local Ollama installation works without external credentials.
            local_default = not self.config.base
            if local_default:
                self.config.base = "http://127.0.0.1:11434/v1"
                if not self.config.key:
                    self.config.key = "ollama"
            if local_default and not any(os.getenv(name) for name in ("LLM_MODEL", "GEN_MODEL", "OPENAI_MODEL")):
                self.config.model = "qwen3:14b"
            # Web defaults; explicit environment settings still take precedence.
            self.config.use_embeddings = os.getenv("USE_EMBEDDINGS", "false").lower() in ("1", "true", "yes", "on")
            self.config.use_rerank = os.getenv("USE_RERANKER", "false").lower() in ("1", "true", "yes", "on")
            self.config.planner = os.getenv("PLANNER", "off")
        path = Path(knowledge_path) if knowledge_path else _ROOT / "knowledge-base"
        self.knowledge = Knowledge.load(path)
        self.retriever = Retriever(self.knowledge)

    def answer(self, messages: list[dict]) -> dict:
        if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
            raise ValueError("Ожидается непустая история сообщений")
        if messages[-1].get("role") != "user" or not isinstance(messages[-1].get("content"), str):
            raise ValueError("Последнее сообщение должно быть от пользователя")
        if not messages[-1]["content"].strip():
            raise ValueError("Вопрос пользователя пуст")
        for item in messages:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant") or not isinstance(item.get("content"), str):
                raise ValueError("Допустимы только текстовые сообщения user и assistant")
        raw = json.dumps(messages, ensure_ascii=False)
        seconds = float(os.getenv("RAG_REQUEST_BUDGET_SECONDS", "90"))
        api = PlatformAPI(self.config, Budget(seconds))
        try:
            result = answer_one(raw, api, self.retriever)
            return {"answer": result.answer, "evidence": result.evidence, "lead": result.lead}
        except APIError as exc:
            raise RAGUnavailableError(f"LLM API недоступен: {exc}") from exc
        finally:
            api.close()
