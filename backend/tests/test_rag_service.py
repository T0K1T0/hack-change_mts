"""Run with: python -m unittest discover -s backend/tests"""
import unittest
import os
from unittest.mock import patch

from backend.app.rag_service import RAGService, RAGUnavailableError
from api_client import Completion


class RAGServiceTest(unittest.TestCase):
    def test_local_ollama_defaults(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "", "LLM_MODEL": "", "LLM_API_KEY": ""}):
            service = RAGService()
        self.assertEqual(service.config.base, "http://127.0.0.1:11434/v1")
        self.assertEqual(service.config.model, "qwen3:14b")
        self.assertFalse(service.config.use_embeddings)
        self.assertFalse(service.config.use_rerank)
        self.assertEqual(service.config.planner, "off")

    def test_dialogue_reaches_existing_answer_pipeline(self):
        seen = []

        def fake_chat(api, messages, **kwargs):
            seen.append(messages[1]["content"])
            return Completion('{"answer":"Для учёта электроэнергии подойдёт АСКУЭ.","evidence":[],"lead":null}', "stop", {})

        with patch("backend.app.rag_service.PlatformAPI.chat", fake_chat):
            service = RAGService()
            result = service.answer([
                {"role": "user", "content": "Нужен учёт электроэнергии."},
                {"role": "assistant", "content": "Уточните тип объекта."},
                {"role": "user", "content": "Для завода нужна АСКУЭ."},
            ])

        self.assertIn("Клиент: Нужен учёт электроэнергии.", seen[0])
        self.assertIn("Консультант: Уточните тип объекта.", seen[0])
        self.assertIn("Клиент: Для завода нужна АСКУЭ.", seen[0])
        self.assertEqual(result, {"answer": "Для учёта электроэнергии подойдёт АСКУЭ.", "evidence": [], "lead": None})

    def test_llm_failure_is_controlled(self):
        from api_client import APIError

        with patch("backend.app.rag_service.PlatformAPI.chat", side_effect=APIError("сеть/тайм-аут")):
            with self.assertRaisesRegex(RAGUnavailableError, "LLM API недоступен"):
                RAGService().answer([{"role": "user", "content": "Что такое АСКУЭ?"}])


if __name__ == "__main__":
    unittest.main()
