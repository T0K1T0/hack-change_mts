"""Live HTTP smoke test against a running Compose deployment.

Run `python -m backend.tests.smoke_http before`, restart the container, then run
`python -m backend.tests.smoke_http after`. Uses the real configured RAG service.
"""
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[2]
MARKER = ROOT / ".browser-test" / "smoke-chat-id"


def env_value(name):
    for line in (ROOT / ".env").read_text(encoding="utf-8-sig").splitlines():
        if line.startswith(name + "="):
            return line.partition("=")[2].strip().strip('"')
    raise RuntimeError(f"Missing {name} in .env")


def check(response, status):
    assert response.status_code == status, (response.request.method, response.request.url, response.status_code, response.text[:500])
    print(f"{response.request.method} {response.request.url.path}: {status}")
    return response.json()


def main(phase):
    port = env_value("HOST_PORT") if "HOST_PORT=" in (ROOT / ".env").read_text(encoding="utf-8-sig") else "8080"
    with httpx.Client(base_url=f"http://localhost:{port}", timeout=160) as client:
        if phase == "before":
            page = client.get("/")
            assert page.status_code == 200 and 'id="auth-screen"' in page.text
            print("GET /: 200, login UI present")
            script = client.get("/frontend.js")
            assert script.status_code == 200 and "showLogin" in script.text
            print("GET /frontend.js: 200")
            check(client.get("/api/health"), 200)
            check(client.get("/api/chats"), 401)
            check(client.post("/api/auth/login", json={"username": env_value("ADMIN_LOGIN"), "password": "definitely-wrong"}), 401)
        login = check(client.post("/api/auth/login", json={"username": env_value("ADMIN_LOGIN"), "password": env_value("ADMIN_PASSWORD")}), 200)
        assert client.cookies.get("energy_session")
        csrf = {"X-CSRF-Token": login["csrf_token"]}
        assert check(client.get("/api/auth/me"), 200)["username"] == env_value("ADMIN_LOGIN")
        if phase == "before":
            check(client.post("/api/chats", json={"title": "QA smoke"}), 403)
            check(client.post("/api/chats", json={}, headers={**csrf, "Origin": "https://other.example"}), 403)
            chat = check(client.post("/api/chats", json={"title": "QA smoke"}, headers=csrf), 201)
            chat_id = chat["id"]
            MARKER.parent.mkdir(exist_ok=True)
            MARKER.write_text(str(chat_id), encoding="ascii")
            reply = client.post(f"/api/chats/{chat_id}/messages", json={"content": "Что такое АСКУЭ?"}, headers=csrf)
            assert reply.status_code in (201, 503), (reply.status_code, reply.text[:500])
            print(f"POST /api/chats/{chat_id}/messages: {reply.status_code}")
            if reply.status_code == 201:
                assert reply.json()["answer"].strip()
            history = check(client.get(f"/api/chats/{chat_id}"), 200)
            assert history["messages"][0]["role"] == "user" and history["messages"][0]["content"] == "Что такое АСКУЭ?"
            assert len(history["messages"]) == (2 if reply.status_code == 201 else 1)
            print("History serialization: OK")
            check(client.get("/api/chats/999999999"), 404)
            check(client.post("/api/auth/logout", headers=csrf), 200)
            check(client.get("/api/auth/me"), 401)
            login = check(client.post("/api/auth/login", json={"username": env_value("ADMIN_LOGIN"), "password": env_value("ADMIN_PASSWORD")}), 200)
            csrf = {"X-CSRF-Token": login["csrf_token"]}
            assert any(c["id"] == chat_id for c in check(client.get("/api/chats"), 200))
            print(f"Preserved chat id for restart check: {chat_id}")
        else:
            chat_id = int(MARKER.read_text(encoding="ascii"))
            history = check(client.get(f"/api/chats/{chat_id}"), 200)
            assert history["messages"][0]["content"] == "Что такое АСКУЭ?"
            print("SQLite survived container restart: OK")
            check(client.delete(f"/api/chats/{chat_id}", headers=csrf), 200)
            check(client.get(f"/api/chats/{chat_id}"), 404)
            check(client.post("/api/auth/logout", headers=csrf), 200)
            check(client.get("/api/auth/me"), 401)
            MARKER.unlink()


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("before", "after"):
        raise SystemExit("Usage: python -m backend.tests.smoke_http before|after")
    main(sys.argv[1])
