# EnergyTool

Демонстрационное веб-приложение AI-консультанта по продуктам и вопросам энергетики. Клиент ведёт диалог в браузере; ответ формируется существующим RAG-кодом на основе Markdown-базы знаний. Это MVP для хакатона.

## Состав проекта

| Путь | Назначение |
| --- | --- |
| `backend/app/app.py` | FastAPI: вход, сессии, чаты, сообщения, healthcheck, отдача интерфейса |
| `backend/app/rag_service.py` | Адаптер между веб-приложением и RAG |
| `backend/app/rag/*.py` | Активные модули RAG: поиск, промпты, правила, LLM-клиент |
| `knowledge-base/**/*.md` | Распакованная база знаний; копируется в Docker-образ |
| `frontend/index.html`, `frontend/frontend.js` | Интерфейс без сборки frontend |
| `backend/tests/test_*.py` | Автоматические тесты; мок LLM только в тестах |
| `backend/tests/smoke_http.py` | Проверка живого HTTP-приложения и сохранности SQLite |

Стек: Python 3.11, FastAPI, Uvicorn, SQLite, Argon2id, httpx, HTML/CSS/JavaScript, Docker Compose. Архивы исходников и базы знаний для работы не нужны.

## Первый запуск

Нужны Docker Desktop или Docker Engine с Compose и доступный OpenAI-совместимый LLM API. Пример `.env.example` настроен на локальный Ollama с моделью `qwen3:14b`. Убедитесь, что модель загружена, а Ollama доступен контейнеру по `host.docker.internal:11434` (при необходимости настройте прослушивание `0.0.0.0:11434`). Для другого провайдера задайте его URL, ключ и модель.

```bash
cp .env.example .env
# Измените ADMIN_PASSWORD и параметры LLM в .env
docker compose config
docker compose up -d --build
docker compose ps
docker compose logs --tail=100
```

Откройте `http://localhost:8080`. Если порт занят, добавьте `HOST_PORT=8081` в `.env` и откройте `http://localhost:8081`. Войдите с `ADMIN_LOGIN` и `ADMIN_PASSWORD` из `.env`. При первом запуске пользователь создаётся автоматически. Изменение пароля в `.env` после первого запуска **не меняет** пароль уже созданного пользователя.

## Переменные окружения

| Переменная | Назначение |
| --- | --- |
| `ADMIN_LOGIN`, `ADMIN_PASSWORD` | Обязательные учётные данные первого администратора |
| `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Адрес, ключ и модель OpenAI-совместимого API; для Ollama ключ `ollama` условный |
| `HOST_PORT` | Порт на хосте, по умолчанию 8080 |
| `COOKIE_SECURE` | `true` при работе за HTTPS; для локального HTTP — `false` |
| `USE_EMBEDDINGS`, `USE_RERANKER`, `PLANNER` | Необязательные этапы RAG; в примере выключены |
| `RAG_REQUEST_BUDGET_SECONDS` | Необязательный лимит времени ответа, по умолчанию 90 секунд |

`DATABASE_PATH=/data/energytool.sqlite3` задан в `backend/Dockerfile`. База хранится в именованном Docker volume `energytool_data` и переживает пересоздание контейнера. Не удаляйте volume, если нужны чаты. Пароль хранится как Argon2id-хеш; браузерная сессия использует HttpOnly cookie и CSRF-токен. `.env` и SQLite исключены из Git и Docker build context.

## Функции и ограничения MVP

Можно войти, создать и удалить чат, отправить сообщение, посмотреть историю и выйти. RAG ищет сведения в `knowledge-base/` и обращается к настроенному LLM API. Если LLM недоступен, сервер возвращает 503 и сохраняет вопрос в истории; подмены ответов фейковым AI в приложении нет.

По явной просьбе клиента или после его согласия консультант может сформировать локальный черновик заявки. Заявка отображается в текущем интерфейсе, но **не сохраняется отдельной записью и не отправляется менеджеру**. После обновления страницы блок заявки не восстанавливается. Интеграции с CRM и сброса пароля через интерфейс нет.

## Проверки

```bash
python -m unittest discover -s backend/tests
python -m backend.tests.smoke_http before
docker compose restart energytool
python -m backend.tests.smoke_http after
```

Для тестов без Docker установите зависимости из `backend/requirements.txt` в виртуальное окружение. Smoke script читает учётные данные из `.env`, отправляет реальные HTTP-запросы к запущенному приложению и использует настроенный RAG. Если LLM недоступен, проверка принимает 503 и проверяет сохранение вопроса. Для остановки приложения: `docker compose down` (данные в volume сохранятся).
