"""EnergyTool MVP API. Start with: uvicorn backend.app.app:app --host 0.0.0.0 --port 8080"""
import asyncio
import hashlib
import os
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .rag_service import RAGService, RAGUnavailableError

ROOT = Path(__file__).resolve().parents[2]
PH = PasswordHasher(time_cost=2, memory_cost=19456, parallelism=1)
COOKIE = "energy_session"
SESSION_SECONDS = 60 * 60 * 24 * 7
MAX_MESSAGE = 4000
MAX_CONTEXT_MESSAGES = 20


class Login(BaseModel):
    username: str
    password: str


class ChatCreate(BaseModel):
    title: str = Field(default="Новый диалог", max_length=120)


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_MESSAGE)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@contextmanager
def connect(path: str):
    db = sqlite3.connect(path, timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=20000")
    try:
        yield db
    finally:
        db.close()


def create_app(db_path: str | None = None, rag_factory=RAGService) -> FastAPI:
    path = db_path or os.getenv("DATABASE_PATH", str(ROOT / "data" / "energytool.sqlite3"))
    attempts: dict[str, list[float]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        username, password = os.getenv("ADMIN_LOGIN"), os.getenv("ADMIN_PASSWORD")
        if not username or not password:
            raise RuntimeError("Задайте ADMIN_LOGIN и ADMIN_PASSWORD")
        with connect(path) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, token_hash TEXT NOT NULL UNIQUE, expires_at INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS chats (id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, title TEXT NOT NULL, created_at INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, created_at INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_chats_user ON chats(user_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
            """)
            if not db.execute("SELECT 1 FROM users WHERE username=?", (username,)).fetchone():
                db.execute("INSERT INTO users(username,password_hash) VALUES (?,?)", (username, PH.hash(password)))
            db.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
            db.commit()
        app.state.rag = rag_factory()
        yield

    app = FastAPI(lifespan=lifespan)

    def current_user(request: Request) -> int:
        token = request.cookies.get(COOKIE, "")
        if not token:
            raise HTTPException(401, "Требуется вход")
        with connect(path) as db:
            row = db.execute("SELECT user_id FROM sessions WHERE token_hash=? AND expires_at>?", (digest(token), int(time.time()))).fetchone()
        if not row:
            raise HTTPException(401, "Сессия истекла")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if not secrets.compare_digest(request.headers.get("X-CSRF-Token", ""), digest("csrf:" + token)):
                raise HTTPException(403, "Неверный CSRF-токен")
            origin = request.headers.get("origin")
            if origin and origin != str(request.base_url).rstrip("/"):
                raise HTTPException(403, "Недопустимый Origin")
        return row["user_id"]

    def owned_chat(chat_id: int, user_id: int):
        with connect(path) as db:
            row = db.execute("SELECT id,user_id,title,created_at FROM chats WHERE id=? AND user_id=?", (chat_id, user_id)).fetchone()
        if not row:
            raise HTTPException(404, "Диалог не найден")
        return dict(row)

    @app.get("/api/health")
    def health():
        with connect(path) as db:
            db.execute("SELECT 1")
        return {"status": "ok"}

    @app.post("/api/auth/login")
    def login(payload: Login, request: Request, response: Response):
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(403, "Недопустимый Origin")
        key = request.client.host if request.client else "unknown"
        now = time.time()
        attempts[key] = [x for x in attempts.get(key, []) if now - x < 300]
        if len(attempts[key]) >= 5:
            raise HTTPException(429, "Слишком много попыток входа. Повторите через 5 минут")
        with connect(path) as db:
            row = db.execute("SELECT id,password_hash FROM users WHERE username=?", (payload.username,)).fetchone()
            try:
                valid = bool(row and PH.verify(row["password_hash"], payload.password))
            except VerifyMismatchError:
                valid = False
            if not valid:
                attempts[key].append(now)
                raise HTTPException(401, "Неверный логин или пароль")
            attempts.pop(key, None)
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO sessions(user_id,token_hash,expires_at) VALUES (?,?,?)", (row["id"], digest(token), int(now) + SESSION_SECONDS))
            db.commit()
        response.set_cookie(COOKIE, token, max_age=SESSION_SECONDS, httponly=True, secure=os.getenv("COOKIE_SECURE", "false").lower() == "true", samesite="lax", path="/")
        return {"username": payload.username, "csrf_token": digest("csrf:" + token)}

    @app.post("/api/auth/logout")
    def logout(request: Request, response: Response, user_id: int = Depends(current_user)):
        with connect(path) as db:
            db.execute("DELETE FROM sessions WHERE token_hash=? AND user_id=?", (digest(request.cookies[COOKIE]), user_id))
            db.commit()
        response.delete_cookie(COOKIE, path="/")
        return {"ok": True}

    @app.get("/api/auth/me")
    def me(request: Request, user_id: int = Depends(current_user)):
        with connect(path) as db:
            row = db.execute("SELECT username FROM users WHERE id=?", (user_id,)).fetchone()
        return {"username": row["username"], "csrf_token": digest("csrf:" + request.cookies[COOKIE])}

    @app.get("/api/chats")
    def list_chats(user_id: int = Depends(current_user)):
        with connect(path) as db:
            return [dict(r) for r in db.execute("SELECT id,title,created_at FROM chats WHERE user_id=? ORDER BY id DESC", (user_id,))]

    @app.post("/api/chats", status_code=201)
    def create_chat(payload: ChatCreate, user_id: int = Depends(current_user)):
        title = payload.title.strip() or "Новый диалог"
        with connect(path) as db:
            cur = db.execute("INSERT INTO chats(user_id,title,created_at) VALUES (?,?,?)", (user_id, title, int(time.time())))
            db.commit()
            return owned_chat(cur.lastrowid, user_id)

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: int, user_id: int = Depends(current_user)):
        chat = owned_chat(chat_id, user_id)
        with connect(path) as db:
            chat["messages"] = [dict(r) for r in db.execute("SELECT id,role,content,created_at FROM messages WHERE chat_id=? ORDER BY id", (chat_id,))]
        return chat

    @app.delete("/api/chats/{chat_id}")
    def delete_chat(chat_id: int, user_id: int = Depends(current_user)):
        with connect(path) as db:
            cur = db.execute("DELETE FROM chats WHERE id=? AND user_id=?", (chat_id, user_id))
            db.commit()
        if not cur.rowcount:
            raise HTTPException(404, "Диалог не найден")
        return {"ok": True}

    @app.post("/api/chats/{chat_id}/messages", status_code=201)
    async def send_message(chat_id: int, payload: MessageCreate, user_id: int = Depends(current_user)):
        content = payload.content.strip()
        if not content:
            raise HTTPException(422, "Сообщение пусто")
        owned_chat(chat_id, user_id)
        with connect(path) as db:
            cur = db.execute("INSERT INTO messages(chat_id,role,content,created_at) VALUES (?,'user',?,?)", (chat_id, content, int(time.time())))
            user_message_id = cur.lastrowid
            db.commit()
            rows = db.execute("SELECT role,content FROM messages WHERE chat_id=? ORDER BY id DESC LIMIT ?", (chat_id, MAX_CONTEXT_MESSAGES)).fetchall()
        history = [dict(r) for r in reversed(rows)]
        try:
            result = await asyncio.to_thread(app.state.rag.answer, history)
        except RAGUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, "Не удалось получить ответ консультанта") from exc
        with connect(path) as db:
            cur = db.execute("INSERT INTO messages(chat_id,role,content,created_at) VALUES (?,'assistant',?,?)", (chat_id, result["answer"], int(time.time())))
            db.commit()
            answer_message_id = cur.lastrowid
        return {"answer": result["answer"], "user_message_id": user_message_id, "answer_message_id": answer_message_id, "evidence": result.get("evidence", []), "lead": result.get("lead")}

    @app.get("/")
    def frontend():
        return FileResponse(ROOT / "frontend" / "index.html")

    @app.get("/frontend.js")
    def frontend_js():
        return FileResponse(ROOT / "frontend" / "frontend.js", media_type="application/javascript")

    return app


app = create_app()
