"""API платформы: ограниченные повторы, разные URL, контроль времени и обрезки.

Единственная внешняя зависимость — httpx. Никаких сторонних LLM-сервисов.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

VERSION = '15'  # единая версия для логов и run_stats.json


def log(text: str) -> None:
    print(f'[EnergyTool v{VERSION}] ' + text, file=sys.stderr, flush=True)


def env(*names: str, default: str = '') -> str:
    for name in names:
        value = os.getenv(name, '').strip()
        if (value and value not in (name, '${' + name + '}', '$' + name)
                and value.lower() not in ('none', 'null', 'undefined')):
            return value
    return default


def number(name: str, default: float, lower: float, upper: float, integer: bool = False):
    try:
        value = float(env(name))
        if not math.isfinite(value):
            raise ValueError('not finite')
        value = min(upper, max(lower, value))
        return int(value) if integer else value
    except (ValueError, TypeError):
        return int(default) if integer else default


def toggle(name: str, default: bool) -> bool:
    value = env(name).lower()
    return default if not value else value in ('1', 'true', 'yes', 'on')


# Defaults are copied from the ORGANIZERS' sample_submission, not guessed aliases.
# Runtime values supplied by the platform always take precedence over these defaults.
DEFAULT_CHAT_MODEL = 'qwen/qwen3.8-27b'
DEFAULT_EMB_MODEL = 'baai/bge-m3'
DEFAULT_RERANK_MODEL = 'qwen/qwen3-reranker-8b'


def wrong_model_family(value: str, role: str) -> bool:
    """Reject recognizable wrong families, but preserve opaque platform aliases.

    Do not require a substring like 'qwen' in the platform's private routing ID.
    A model's actual identity cannot be proven just by inspecting that ID.
    """
    name = value.lower()
    if re.search(r'(?:^|[/.:_-])(?:gpt[-_]|chatgpt|text-embedding-|o[134](?:[-_]|$))', name):
        return True
    if role == 'chat':
        return any(word in name for word in ('bge-m3', 'reranker', 'embedding'))
    if role == 'embeddings':
        return any(word in name for word in ('reranker', 'qwen3.8', 'qwen3-27b'))
    return any(word in name for word in ('bge-m3', 'embedding', 'qwen3.8'))


def resolve_model(names: tuple[str, ...], fallback: str, role: str) -> tuple[str, str, list[str]]:
    notices = []
    for name in names:
        value = env(name)
        if not value:
            continue
        # ID маршрута может быть непрозрачным. Регламент требует использовать
        # именно значение платформы, а не подменять его по имени семейства.
        return value, name, notices
    notices.append(f'{role}: идентификатор не передан; используется значение из примера организаторов.')
    return fallback, 'organizer_sample_default', notices


@dataclass
class Config:
    base: str
    key: str
    model: str
    workers: int = 4
    context_chars: int = 20000
    max_tokens: int = 1200
    temperature: float = 0.1
    runtime: float = 1100.0
    request_timeout: float = 65.0
    planner: str = 'off'
    emb_base: str = ''
    emb_key: str = ''
    emb_model: str = 'baai/bge-m3'
    rerank_base: str = ''
    rerank_key: str = ''
    rerank_model: str = 'qwen/qwen3-reranker-8b'
    use_embeddings: bool = False
    use_rerank: bool = False
    model_source: str = 'explicit_config'
    model_notices: list[str] = field(default_factory=list)

    @classmethod
    def load(cls) -> 'Config':
        base = env('LLM_BASE_URL', 'BASE_URL', 'OPENAI_BASE_URL', 'API_BASE_URL', 'OPENAI_API_BASE')
        key = env('LLM_API_KEY', 'OPENAI_API_KEY', 'API_KEY')
        emb_key = env('EMBEDDER_API_KEY', 'EMBEDDING_API_KEY', 'EMB_API_KEY')
        emb_base = env('EMBEDDER_BASE_URL', 'EMBEDDING_BASE_URL', 'EMBEDDINGS_BASE_URL', 'EMB_BASE_URL')
        rank_key = env('RERANKER_API_KEY', 'RERANK_API_KEY')
        rank_base = env('RERANKER_BASE_URL', 'RERANK_BASE_URL')
        # Регламент: все три модели доступны через ОБЩИЙ локальный прокси.
        # Отдельные адреса/ключи для embeddings и rerank НЕ обязательны.
        use_emb = toggle('USE_EMBEDDINGS', bool(base or emb_base))
        use_rank = toggle('USE_RERANKER', bool(base or rank_base))
        model, model_source, notices = resolve_model(
            ('LLM_MODEL', 'GEN_MODEL', 'OPENAI_MODEL'), DEFAULT_CHAT_MODEL, 'chat')
        emb_model, _, emb_notices = resolve_model(
            ('EMB_MODEL', 'EMBEDDER_MODEL', 'EMBEDDING_MODEL'), DEFAULT_EMB_MODEL, 'embeddings')
        rerank_model, _, rank_notices = resolve_model(
            ('RERANKER_MODEL', 'RERANK_MODEL'), DEFAULT_RERANK_MODEL, 'rerank')
        unused = [name for name in ('CHAT_MODEL', 'REVIEW_MODEL', 'PLAN_MODEL') if env(name)]
        if unused:
            notices.append(', '.join(unused) + ': не используются; все текстовые этапы работают через LLM_MODEL/GEN_MODEL/OPENAI_MODEL.')
        return cls(
            base=base.rstrip('/'), key=key, model=model,
            model_source=model_source, model_notices=notices + emb_notices + rank_notices,
            workers=number('MAX_WORKERS', 4, 1, 8, True),
            context_chars=number('MAX_CONTEXT_CHARS', 22000, 4000, 40000, True),
            max_tokens=number('MAX_TOKENS', 1600, 256, 8192, True),
            temperature=number('TEMPERATURE', 0.1, 0, 1),
            runtime=number('RUN_TIMEOUT_SECONDS', 1100, 10, 1150),
            request_timeout=number('REQUEST_TIMEOUT', 65, 5, 120),
            planner=env('PLANNER', default='reactive'),
            emb_base=(emb_base or base).rstrip('/'), emb_key=emb_key or key,
            emb_model=emb_model,
            rerank_base=(rank_base or base).rstrip('/'), rerank_key=rank_key or key,
            rerank_model=rerank_model,
            use_embeddings=use_emb, use_rerank=use_rank,
        )


class Budget:
    def __init__(self, seconds: float, questions: int = 1, workers: int = 1):
        self.deadline = time.monotonic() + seconds
        self.questions = questions
        self.workers = workers
        self.completed = 0
        self.average_call = 12.0
        self.lock = threading.Lock()

    def remaining(self) -> float:
        return max(0.0, self.deadline-time.monotonic())

    def optional(self, multiplier: float = 1.0) -> bool:
        with self.lock:
            pending = max(1, self.questions - self.completed)
            reserve = (pending / max(1, self.workers)) * max(8, self.average_call) * 1.3
        return self.remaining() > reserve + max(25, multiplier*self.average_call)

    def observed(self, elapsed: float) -> None:
        with self.lock:
            self.average_call = 0.8*self.average_call + 0.2*min(120, elapsed)

    def done(self) -> None:
        with self.lock:
            self.completed += 1


class APIError(RuntimeError):
    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


class ContextTooLong(APIError):
    pass


class TruncatedAnswer(APIError):
    pass


@dataclass
class Completion:
    content: str
    finish_reason: str
    usage: dict


def endpoint_candidates(base: str, endpoint: str) -> list[str]:
    base = base.rstrip('/')
    if not base:
        return []
    if base.endswith('/' + endpoint):
        return [base]
    # Также принимает LLM_BASE_URL, заканчивающийся готовым chat endpoint.
    for suffix in ('/chat/completions', '/embeddings', '/rerank'):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    if base.endswith('/v1'):
        return [base + '/' + endpoint]
    return [base + '/' + endpoint, base + '/v1/' + endpoint]


def strip_thinking(text: str) -> str:
    text = re.sub(r'<think\b[^>]*>.*?</think\s*>', '', text, flags=re.I | re.S)
    if re.search(r'<think\b', text, re.I):
        text = re.sub(r'<think\b.*$', '', text, flags=re.I | re.S)
    # Некоторые шаблоны возвращают только закрывающую метку.
    if re.search(r'</think\s*>', text, re.I):
        text = re.split(r'</think\s*>', text, flags=re.I)[-1]
    return text.strip()


def parse_json(text: str) -> dict:
    text = strip_thinking(text).strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text, flags=re.I)
    decoder = json.JSONDecoder()
    start = text.find('{')
    if start < 0:
        raise ValueError('Ожидался JSON-объект')
    # Never substitute a nested lead object for an unfinished outer answer.
    try:
        value, _ = decoder.raw_decode(text[start:])
    except ValueError:
        raise ValueError('Незавершённый JSON-объект') from None
    if not isinstance(value, dict):
        raise ValueError('Ожидался JSON-объект')
    return value



def recover_completed_answer(text: str) -> dict | None:
    """Recover only fully decoded TOP-LEVEL fields from a cut JSON object.

    An unfinished answer string, inner lead.answer, or arbitrary free text cannot
    be salvaged. Lead is deliberately dropped unless the whole JSON is valid.
    """
    text = strip_thinking(text).strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.I)
    start = text.find('{')
    if start < 0:
        return None
    decoder = json.JSONDecoder()
    pos, fields = start + 1, {}
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        try:
            key, pos = decoder.raw_decode(text, pos)
        except ValueError:
            break
        if not isinstance(key, str):
            break
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != ':':
            break
        pos += 1
        while pos < len(text) and text[pos].isspace():
            pos += 1
        try:
            value, pos = decoder.raw_decode(text, pos)
        except ValueError:
            break
        fields[key] = value
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != ',':
            break
        pos += 1
    answer = fields.get('answer')
    if not isinstance(answer, str) or not answer.strip():
        return None
    # A completed JSON string may itself contain a cut sentence. In a cut
    # object require a normal sentence ending; prefer a retry otherwise.
    if not re.search(r'[.!?…»"”)]\s*$', answer):
        return None
    return {'answer': answer, 'evidence': fields.get('evidence', []),
            'lead': None, 'recovered': True}


class PlatformAPI:
    def __init__(self, config: Config, budget: Budget, client: httpx.Client | None = None):
        self.cfg, self.budget = config, budget
        self.client = client or httpx.Client(
            limits=httpx.Limits(max_connections=config.workers+4, max_keepalive_connections=config.workers+4),
            follow_redirects=False,
        )
        self.owns_client = client is None
        self.selected_urls = {}
        self.profile = 0  # 0: vLLM, 1: enable_thinking, 2: без дополнительных полей
        self.token_parameter = 'max_tokens'
        self.token_cap = 8192
        self.disabled = set()
        self.fatal_chat = False
        self.lock = threading.Lock()
        self.stats = CounterSafe()

    def close(self) -> None:
        if self.owns_client:
            self.client.close()

    def _request(self, kind: str, body: dict, timeout: float | None = None) -> dict:
        if self.budget.remaining() < 3:
            raise APIError('Исчерпан бюджет выполнения')
        if kind in self.disabled or (kind == 'chat' and self.fatal_chat):
            raise APIError('Сервис отключен после ошибки конфигурации')
        if kind == 'chat' and not self.cfg.model:
            self.fatal_chat = True
            raise APIError('Не задан идентификатор модели в LLM_MODEL/GEN_MODEL/OPENAI_MODEL')
        expected_model = {'chat': self.cfg.model, 'embeddings': self.cfg.emb_model,
                          'rerank': self.cfg.rerank_model}.get(kind)
        if body.get('model') != expected_model:
            raise APIError(f'{kind}: модель не соответствует конфигурации платформы')
        if kind == 'chat':
            base, key, endpoint = self.cfg.base, self.cfg.key, 'chat/completions'
        elif kind == 'embeddings':
            base, key, endpoint = self.cfg.emb_base or self.cfg.base, self.cfg.emb_key or self.cfg.key, 'embeddings'
        else:
            base, key, endpoint = self.cfg.rerank_base or self.cfg.base, self.cfg.rerank_key or self.cfg.key, 'rerank'
        urls = endpoint_candidates(base, endpoint)
        if not urls:
            raise APIError('Не передан адрес API платформы')
        if kind in self.selected_urls:
            urls = [self.selected_urls[kind]] + [u for u in urls if u != self.selected_urls[kind]]
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer ' + key
        payload = dict(body)
        transient = 0
        url_index = 0
        # Общий предел включает проверку URL и снятие несовместимых параметров.
        for _ in range(8):
            left = self.budget.remaining() - 1
            if left < 1:
                raise APIError('Исчерпан бюджет запроса')
            request_time = min(timeout or self.cfg.request_timeout, left)
            started = time.monotonic()
            self.stats.add(kind + '_requests')
            try:
                response = self.client.post(
                    urls[url_index], json=payload, headers=headers,
                    timeout=httpx.Timeout(request_time, connect=min(10, request_time), pool=min(10, request_time)),
                )
                if kind == 'chat':
                    self.budget.observed(time.monotonic() - started)
            except (httpx.TimeoutException, httpx.TransportError):
                if kind == 'chat':
                    self.budget.observed(time.monotonic() - started)
                transient += 1
                self.stats.add(kind + '_transport_errors')
                if transient >= (2 if kind == 'chat' else 1) or self.budget.remaining() < 5:
                    raise APIError(f'{kind}: сеть/тайм-аут') from None
                time.sleep(min(transient, max(0, self.budget.remaining()-2)))
                continue
            code = response.status_code
            if code in (401, 403):
                if kind == 'chat':
                    self.fatal_chat = True
                else:
                    self.disabled.add(kind)
                raise APIError(f'{kind}: доступ отклонён (HTTP {code})', code)
            if code in (404, 405) and url_index + 1 < len(urls) and 'model' not in response.text.lower():
                url_index += 1
                continue
            if kind == 'chat' and code in (400, 422, 500) and any(
                    word in response.text.lower() for word in ('enable_thinking', 'chat_template_kwargs')):
                if 'chat_template_kwargs' in payload or 'enable_thinking' in payload:
                    payload.pop('chat_template_kwargs', None)
                    payload.pop('enable_thinking', None)
                    self.profile = 2
                    self.stats.add('parameter_negotiations')
                    continue
            if code in (408, 409, 425, 429, 500, 502, 503, 504):
                transient += 1
                self.stats.add(kind + '_retries')
                if transient >= (2 if kind == 'chat' else 1):
                    raise APIError(f'{kind}: HTTP {code} после повторов', code)
                try:
                    delay = max(0.1, min(10.0, float(response.headers.get('Retry-After', transient))))
                except ValueError:
                    delay = float(transient)
                if self.budget.remaining() < delay + 3:
                    raise APIError(f'{kind}: недостаточно времени для повтора', code)
                time.sleep(delay)
                continue
            if code in (400, 413, 422):
                error = response.text.lower()
                if any(x in error for x in ('context length', 'context_length', 'context window', 'too many tokens', 'input is too long', 'prompt is too long')) or code == 413:
                    raise ContextTooLong('API: слишком большой контекст', code)
                # Поддержка ограничений сервера на размер генерации.
                match = re.search(r'(?:max_tokens|max_completion_tokens).{0,100}?(?:less than or equal to|at most|maximum(?: is)?|<=|must be <=)\s*(\d{3,5})', error)
                if match:
                    cap = max(128, int(match.group(1)))
                    self.token_cap = min(self.token_cap, cap)
                    for parameter in ('max_tokens', 'max_completion_tokens'):
                        if parameter in payload and payload[parameter] > cap:
                            payload[parameter] = cap
                            break
                    else:
                        raise APIError(f'{kind}: неподдерживаемые параметры HTTP {code}', code)
                    continue
                if kind == 'chat' and 'chat_template_kwargs' in payload:
                    payload.pop('chat_template_kwargs', None)
                    payload['enable_thinking'] = False
                    self.profile = 1
                    self.stats.add('parameter_negotiations')
                    continue
                if kind == 'chat' and 'enable_thinking' in payload:
                    payload.pop('enable_thinking', None)
                    self.profile = 2
                    self.stats.add('parameter_negotiations')
                    continue
                if kind == 'chat' and 'max_tokens' in payload and 'max_completion_tokens' in error:
                    payload['max_completion_tokens'] = payload.pop('max_tokens')
                    self.token_parameter = 'max_completion_tokens'
                    continue
                raise APIError(f'{kind}: параметры не приняты (HTTP {code})', code)
            if code >= 300:
                raise APIError(f'{kind}: HTTP {code}', code)
            try:
                data = response.json()
            except ValueError:
                raise APIError(f'{kind}: невалидный JSON ответа') from None
            if not isinstance(data, dict):
                raise APIError(f'{kind}: неверный формат ответа')
            self.selected_urls[kind] = urls[url_index]
            return data
        raise APIError(f'{kind}: исчерпан предел согласования API')

    def chat(self, messages: list[dict], max_tokens: int | None = None,
             purpose: str = 'answer') -> Completion:
        requested = max_tokens or self.cfg.max_tokens
        self.stats.add('calls_' + purpose)
        for attempt in range(2):
            cap = min(requested * (2**attempt), self.token_cap, 8192)
            body = {'model': self.cfg.model, 'messages': messages,
                    'temperature': self.cfg.temperature if purpose == 'answer' else 0.1,
                    self.token_parameter: cap, 'stream': False}
            if self.profile == 0:
                body['chat_template_kwargs'] = {'enable_thinking': False}
            elif self.profile == 1:
                body['enable_thinking'] = False
            data = self._request('chat', body, timeout=min(18, self.cfg.request_timeout) if purpose == 'planner' else None)
            choices = data.get('choices')
            if not isinstance(choices, list) or not choices:
                if attempt == 0 and self.budget.remaining() > 10:
                    continue
                raise APIError('API не вернул choices')
            choice = choices[0]
            if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
                self.stats.add('malformed_completions')
                if attempt == 0 and self.budget.remaining() > 10:
                    continue
                raise APIError('API вернул неверную структуру message')
            message = choice['message']
            content = message.get('content') or ''
            if isinstance(content, list):
                content = '\n'.join(item['text'] for item in content
                                    if isinstance(item, dict) and isinstance(item.get('text'), str))
            content = strip_thinking(content) if isinstance(content, str) else ''
            # Reasoning is never a substitute for a final answer or JSON result.
            if not content and (message.get('reasoning_content') or message.get('reasoning')):
                self.stats.add('reasoning_only_completions')
            finish = str(choice.get('finish_reason') or '')
            if finish == 'length':
                self.stats.add('truncated_completions')
                # Обычный ответ не обязан быть JSON. Его завершённость и сохранение
                # полезной части контролирует solution.py, не клиент транспорта.
                if purpose in ('lead', 'planner'):
                    try:
                        parse_json(content)
                        self.stats.add('complete_json_at_token_limit')
                    except ValueError:
                        recovered = recover_completed_answer(content)
                        if recovered is not None and purpose == 'lead':
                            content = json.dumps(recovered, ensure_ascii=False)
                            self.stats.add('salvaged_complete_answers')
                        elif attempt == 0 and cap < self.token_cap and self.budget.remaining() > 12:
                            continue
                        else:
                            raise TruncatedAnswer('Генерация оборвалась до завершения JSON') from None
            if content:
                usage = data.get('usage') or {}
                usage = usage if isinstance(usage, dict) else {}
                for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    if isinstance(usage.get(name), int):
                        self.stats.add(name, usage[name])
                return Completion(content, finish, usage)
            self.stats.add('empty_completions')
            if self.budget.remaining() < 12:
                break
        raise APIError('API не вернул текст ответа; рассуждения не выдаются за ответ')

    def embeddings(self, texts: list[str]) -> list[list[float]]:
        data = self._request('embeddings', {'model': self.cfg.emb_model, 'input': texts}, timeout=min(18, self.cfg.request_timeout))
        rows = data.get('data', [])
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise APIError('Неполный пакет эмбеддингов')
        ordered = {}
        for position, row in enumerate(rows):
            idx = int(row.get('index', position))
            if idx in ordered:
                raise APIError('Повторный индекс эмбеддинга')
            vector = [float(x) for x in row.get('embedding', [])]
            if not vector or not all(math.isfinite(x) for x in vector):
                raise APIError('Некорректный эмбеддинг')
            norm = math.sqrt(sum(x*x for x in vector))
            if norm == 0:
                raise APIError('Нулевой эмбеддинг')
            ordered[idx] = [x / norm for x in vector]
        if set(ordered) != set(range(len(texts))):
            raise APIError('Нарушен порядок индексов эмбеддингов')
        vectors = [ordered[i] for i in range(len(texts))]
        if len({len(v) for v in vectors}) != 1:
            raise APIError('Разные размерности эмбеддингов')
        return vectors

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        data = self._request('rerank', {'model': self.cfg.rerank_model, 'query': query,
                                       'documents': documents, 'top_n': len(documents)}, timeout=min(18, self.cfg.request_timeout))
        rows = data.get('results', data.get('data', []))
        if not isinstance(rows, list):
            raise APIError('Неверный формат rerank')
        # Не доверяем порядку массива, если сервер отдал оценки без сортировки.
        scored = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                score = float(row.get('relevance_score', row.get('score', 0)))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score):
                scored.append((score, row))
        rows = [row for score, row in sorted(scored, key=lambda pair: -pair[0])]
        result = []
        for row in rows:
            raw = row.get('index')
            # int(1.9) silently selects the wrong document; bool is also an int.
            if isinstance(raw, bool) or not isinstance(raw, (int, str)):
                continue
            if isinstance(raw, str) and not re.fullmatch(r'[0-9]+', raw):
                continue
            try:
                idx = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 0 <= idx < len(documents) and idx not in result:
                result.append(idx)
        if not result:
            raise APIError('Пустой результат rerank')
        return result


class CounterSafe:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def add(self, key: str, amount: int = 1) -> None:
        with self.lock:
            self.values[key] = self.values.get(key, 0) + amount

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.values)
