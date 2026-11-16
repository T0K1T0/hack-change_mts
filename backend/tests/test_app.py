import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.app import PH, connect, create_app
from backend.app.rag_service import RAGUnavailableError


class FakeRAG:
    def answer(self, messages):
        return {"answer": "Ответ: " + messages[-1]["content"], "evidence": ["D01S01"], "lead": None}


class UnavailableRAG:
    def answer(self, messages):
        raise RAGUnavailableError("LLM API недоступен")


class APITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "test.sqlite3")
        self.env = patch.dict(os.environ, {"ADMIN_LOGIN": "admin", "ADMIN_PASSWORD": "correct horse battery staple"})
        self.env.start()
        self.client = TestClient(create_app(self.db, FakeRAG))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.env.stop()
        self.tmp.cleanup()

    def login(self, username="admin", password="correct horse battery staple"):
        result = self.client.post("/api/auth/login", json={"username": username, "password": password})
        self.assertEqual(result.status_code, 200)
        return result.json()["csrf_token"]

    def test_passwords_and_session(self):
        self.assertEqual(self.client.post("/api/auth/login", json={"username": "admin", "password": "wrong"}).status_code, 401)
        csrf = self.login()
        with connect(self.db) as db:
            self.assertNotEqual(db.execute("SELECT password_hash FROM users").fetchone()[0], "correct horse battery staple")
            self.assertNotEqual(db.execute("SELECT token_hash FROM sessions").fetchone()[0], self.client.cookies.get("energy_session"))
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)
        self.assertEqual(self.client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

    def test_auth_chats_messages_and_csrf(self):
        self.assertEqual(self.client.get("/api/chats").status_code, 401)
        csrf = self.login()
        self.assertEqual(self.client.post("/api/chats", json={}).status_code, 403)
        headers = {"X-CSRF-Token": csrf}
        created = self.client.post("/api/chats", json={"title": "Завод"}, headers=headers)
        self.assertEqual(created.status_code, 201)
        chat_id = created.json()["id"]
        reply = self.client.post(f"/api/chats/{chat_id}/messages", json={"content": "Что такое АСКУЭ?"}, headers=headers)
        self.assertEqual(reply.status_code, 201)
        self.assertEqual(reply.json()["answer"], "Ответ: Что такое АСКУЭ?")
        self.assertEqual(len(self.client.get(f"/api/chats/{chat_id}").json()["messages"]), 2)
        self.assertEqual(len(self.client.get("/api/chats").json()), 1)

    def test_other_user_cannot_access_chat(self):
        csrf = self.login()
        chat_id = self.client.post("/api/chats", json={}, headers={"X-CSRF-Token": csrf}).json()["id"]
        with connect(self.db) as db:
            db.execute("INSERT INTO users(username,password_hash) VALUES (?,?)", ("other", PH.hash("other-password")))
            db.commit()
        self.client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf})
        other_csrf = self.login("other", "other-password")
        self.assertEqual(self.client.get(f"/api/chats/{chat_id}").status_code, 404)
        self.assertEqual(self.client.delete(f"/api/chats/{chat_id}", headers={"X-CSRF-Token": other_csrf}).status_code, 404)
        self.assertEqual(self.client.post(f"/api/chats/{chat_id}/messages", json={"content": "test"}, headers={"X-CSRF-Token": other_csrf}).status_code, 404)

    def test_data_survives_restart(self):
        csrf = self.login()
        chat_id = self.client.post("/api/chats", json={}, headers={"X-CSRF-Token": csrf}).json()["id"]
        self.client.post(f"/api/chats/{chat_id}/messages", json={"content": "Сохранить"}, headers={"X-CSRF-Token": csrf})
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(self.db, FakeRAG))
        self.client.__enter__()
        self.login()
        self.assertEqual(len(self.client.get(f"/api/chats/{chat_id}").json()["messages"]), 2)

    def test_llm_failure_returns_503_and_saves_question(self):
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(self.db, UnavailableRAG))
        self.client.__enter__()
        csrf = self.login()
        headers = {"X-CSRF-Token": csrf}
        chat_id = self.client.post("/api/chats", json={}, headers=headers).json()["id"]
        response = self.client.post(f"/api/chats/{chat_id}/messages", json={"content": "Вопрос"}, headers=headers)
        self.assertEqual(response.status_code, 503)
        messages = self.client.get(f"/api/chats/{chat_id}").json()["messages"]
        self.assertEqual([(item["role"], item["content"]) for item in messages], [("user", "Вопрос")])


if __name__ == "__main__":
    unittest.main()
