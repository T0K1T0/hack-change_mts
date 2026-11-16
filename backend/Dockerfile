FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app/ ./backend/app/
COPY backend/__init__.py ./backend/__init__.py
COPY frontend/ ./frontend/
COPY knowledge-base/ ./knowledge-base/
ENV PYTHONUNBUFFERED=1 DATABASE_PATH=/data/energytool.sqlite3
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=3)"
CMD ["uvicorn", "backend.app.app:app", "--host", "0.0.0.0", "--port", "8080"]
