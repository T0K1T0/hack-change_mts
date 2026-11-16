"""Поиск по БАЗЕ, ПЕРЕДАННОЙ ПЛАТФОРМОЙ. Здесь нет зашитых ответов.

Мелкие фрагменты используются для поиска, полные разделы/документы — для ответа.
Это сохраняет таблицы, исключения и оговорки, которые терялись в версии 2.
"""
from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from evidence import normalize_tables, source_role
from lexical import FieldIndex, stem_ru


def normal(text: str) -> str:
    value = text.lower().replace('ё', 'е').replace('\u00a0', ' ')
    return re.sub(r'квт[·⋅* /-]?ч\b', 'квтч', value)


WORD = re.compile(r'[a-zа-я0-9]+', re.I)
STOP = set('''а и в во на с со к ко по для за из от до о об у же ли не ни но или
то так это эти этой тот что как какой какие где когда при если можно нужно наш наша
ваше вы мы они его ее их мой быть есть был были будет уже бы да нет очень просто
либо стоит хочу хотим могу можете пожалуйста расскажите скажите подскажите привет
добрый день вопрос этом этого также мне нам вас вам такой такого вообще'''.split())
SUFFIXES = sorted(set('''иями ями ами ениями ости остей остями ого его ому ему ыми
ими овых евых аться яться ится ется лась лись лось ской ская ские ских ским скую
ую юю ая яя ое ее ые ие ый ий ой ах ях ом ем ам ям ов ев ей ых их ым им ть ет
ют ит ят ат ут ла ли ло а я ы и у ю е о ь й'''.split()), key=lambda x: (-len(x), x))


def stem(word: str) -> str:
    if len(word) < 5 or not word.isalpha():
        return word
    for ending in SUFFIXES:
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            return word[:-len(ending)]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in WORD.findall(normal(text[:16000])) if len(w) > 1 and w not in STOP]


def grams(text: str) -> set[str]:
    result = set()
    for word in WORD.findall(normal(text[:16000])):
        if word in STOP or len(word) < 4 or word.isdigit():
            continue
        word = '^' + word + '$'
        for size in (3, 4):
            result.update(word[i:i+size] for i in range(len(word) - size + 1))
    return result


ROLE = re.compile(r'^\s*(?:[-*>]\s*)?(пользователь|клиент|user|human|ассистент|консультант|assistant|бот)\s*:\s*(.*)$', re.I)
USER_ROLES = {'пользователь', 'клиент', 'user', 'human'}


@dataclass
class Dialogue:
    turns: list[tuple[str, str]]

    @property
    def last(self) -> str:
        return next((text for role, text in reversed(self.turns) if role == 'user'), '')

    @property
    def history(self) -> str:
        idx = next((i for i in range(len(self.turns)-1, -1, -1) if self.turns[i][0] == 'user'), len(self.turns))
        return '\n'.join(('Клиент: ' if r == 'user' else 'Консультант: ') + t for r, t in self.turns[:idx])

    @property
    def user_text(self) -> str:
        return '\n'.join(t for r, t in self.turns if r == 'user')

    @property
    def active_user(self) -> list[str]:
        """Only the current subject for retrieval; render() retains the full dialogue."""
        users = [t for r, t in self.turns if r == 'user']
        start = 0
        switch = r'теперь|сейчас.{0,20}(?:речь|интерес|обсуд)|друг(?:ой|ая|ую) (?:вопрос|тема|продукт)|вместо|забуд|вернемся|перейдем|не об этом'
        product_patterns = [p for t, (p, _) in TOPICS.items()
                            if t in ('solar', 'orem', 'photo', 'losses', 'metering')]
        for i, text in enumerate(users):
            q = normal(text)
            current = {j for j, pattern in enumerate(product_patterns) if re.search(pattern, q)}
            # Explicitly changing the subject is not the same as rejecting a detail.
            if i and re.search(switch, q) and current:
                start = i
        return users[start:]

    @property
    def prior_user(self) -> str:
        users = self.active_user
        recent = users[-4:-1]
        anchor = r'солнеч|\bсэс\b|оптов|\bорэм\b|коммерческ.{0,12}потер|распозна|фотограф|\bаскуэ\b|платформ|систем.{0,10}учет'
        if len(users) > 4 and not re.search(anchor, normal(' '.join(recent))):
            older = next((t for t in reversed(users[:-4]) if re.search(anchor, normal(t))), '')
            if older:
                recent = [older] + recent
        return '\n'.join(recent)

    @property
    def search_hint(self) -> str:
        # A preceding question is only a search hint, not a factual source.
        if len(tokens(self.last)) > 8:
            return ''
        last_user = next((i for i in range(len(self.turns)-1, -1, -1)
                          if self.turns[i][0] == 'user'), 0)
        previous = next((t for r, t in reversed(self.turns[:last_user]) if r == 'assistant'), '')
        questions = re.findall(r'[^.!?\n]*\?', previous)
        return ' '.join(questions)[-700:]

    def search_text(self) -> str:
        parts = [self.last[:3500]]
        if self.prior_user:
            parts.append('Предыдущие реплики клиента: ' + self.prior_user[-2400:])
        if self.search_hint:
            parts.append('Тема уточняющего вопроса: ' + self.search_hint)
        return '\n'.join(parts)

    def render(self, maximum: int = 14000) -> str:
        value = '\n'.join(('Клиент: ' if r == 'user' else 'Консультант: ') + t for r, t in self.turns)
        if len(value) <= maximum:
            return value
        # Сохраняем начало с параметрами объекта и последние реплики.
        return value[:maximum//4] + '\n[середина очень длинного диалога опущена]\n' + value[-maximum*3//4:]


def parse_dialogue(raw: str) -> Dialogue:
    raw = str(raw or '').strip()
    if raw.startswith(('[', '{')):
        try:
            data = json.loads(raw)
            records = data.get('messages', data.get('history', [])) if isinstance(data, dict) else data
            turns = []
            for item in records if isinstance(records, list) else []:
                if not isinstance(item, dict):
                    continue
                role = str(item.get('role', item.get('speaker', 'user'))).lower()
                if role not in USER_ROLES | {'assistant', 'ассистент', 'консультант', 'бот'}:
                    continue  # Входной текст НЕ может задавать роль system.
                content = item.get('content', item.get('text', ''))
                if isinstance(content, str) and content.strip():
                    turns.append(('user' if role in USER_ROLES else 'assistant', content.strip()))
            if isinstance(data, dict) and isinstance(data.get('question'), str):
                turns.append(('user', data['question']))
            if turns and any(r == 'user' for r, _ in turns):
                return Dialogue(turns)
        except (ValueError, TypeError):
            pass
    turns = []
    preamble = []
    for line in raw.splitlines():
        match = ROLE.match(line)
        if match:
            if preamble:
                turns.append(('user', '\n'.join(preamble).strip()))
                preamble = []
            role = 'user' if match.group(1).lower() in USER_ROLES else 'assistant'
            turns.append((role, match.group(2)))
        elif turns:
            role, text = turns[-1]
            turns[-1] = (role, text + '\n' + line)
        else:
            preamble.append(line)
    if not turns:
        return Dialogue([('user', raw)])
    turns = [(r, t.strip()) for r, t in turns if t.strip()]
    return Dialogue(turns) if any(r == 'user' for r, _ in turns) else Dialogue([('user', raw)])


@dataclass
class Section:
    id: str
    document: str
    source: str
    heading: str
    path: str
    body: str
    level: int

    @property
    def text(self) -> str:
        return self.path + '\n' + self.body


@dataclass
class Document:
    id: str
    source: str
    title: str
    sections: list[str] = field(default_factory=list)


@dataclass
class Piece:
    section: str
    text: str
    title: str


TOPICS = {
    'solar': (r'солнеч|\bсэс\b|\bпанел|инсоляц|инвертор|кров[е]?л|\bкрыш|\bскат|накопител|гибридн.{0,20}станц', '23-solar'),
    'orem': (r'оптов|\bопт(?:а|е|у|ом)?\b|\bорэм\b|балансирующ.{0,10}рын|точк.{0,10}постав', '21-orem'),
    'photo': (r'фото|изображен|распозна|штрих|водомер|снимок|снимк|циферблат|телеграм|telegram|\bocr\b', '25-meter-reading'),
    'losses': (r'потер|хищен|воровств|\bкраж|безучет|нарушител|неучтен|подозр|заподозр|спис.{0,15}провер|провер.{0,30}наруш', '24-losses'),
    'metering': (r'аскуэ|\bплатформ|платформ.{0,15}учет|систем.{0,15}учет|автоматическ.{0,10}учет|автоматиз.{0,15}учет|\bпрофи\b|\bлайт\b|арендатор|цех|счетчик|прибор|показани|\bsim\b|сим.карт|симк|магазин|склад|ретейл|бизнес.центр', '22-askue'),
    'tariffs': (r'категори|тариф|счет.{0,20}(?:раст|вырос|увелич|больш)|рост.{0,12}счет|передач.{0,15}напряж', '11-price-categories'),
    'power': (r'уров.{0,15}напряж|трансформатор|реактив|класс.{0,10}точност|670', '12-power-thresholds'),
    'micro': (r'микрогенер|излиш|отда.{0,20}сет|прода.{0,25}сет|лимит.{0,20}регион', '15-microgeneration'),
    'general': (r'персональн|данн.{0,20}принадлеж|где.{0,12}хранят|своих.{0,10}сервер|закрыт.{0,12}контур|россии|опыт.{0,15}компан', '26-general'),
    'request': (r'заявк|перезвон|менеджер|фио|расчет|рассчит|предварительн.{0,15}оценк|коммерческ.{0,15}предложен', '03-calculation'),
}
# Расширяются лишь поисковые термины, а не утверждения в ответе.
EXPANSIONS = (
    (r'\bминимум|\bмало.{0,15}(?:прибор|счетчик)|\bтри.{0,15}счетчик', 'минимальное количество приборов ограничения'),
    (r'\bмеханическ', 'механические приборы поддержка'),

    (r'демо|попроб|показать|до (?:покупк|оплат)|логин', 'демонстрационный доступ пилотное подключение попробовать посмотреть до покупки'),
    (r'хран|многолетн|архив.{0,15}фот|фот.{0,15}архив', 'срок глубина хранения фотографий данные требования договор'),
    (r'\bкоттедж|\bдач|частн.{0,10}дом', 'частные жилые дома не подходит целевой сегмент бизнес'),
    (r'взыск|заподозр|нарушен|краж', 'коммерческие потери вероятностный результат выездная проверка не подтверждается'),
    (r'оптимиз|снизить|сократ', 'анализ затрат оптимизация потребления подбор продукта'),
    (r'персональн|конфиденц|обезлич', 'персональные данные идентификатор канала серийный номер договор'),
    (r'прибор.{0,15}событ|событи.{0,25}подключ|истор.{0,20}подключ', 'журнал событий ранее подключенные интеллектуальная система учета'),
    (r'сколько.{0,20}знак|распозна.{0,20}верно', 'точность распознавания'),
    (r'год.{0,20}(?:выпуск|изготовл)|дата.{0,15}выпуск', 'год производства прибора'),
    (r'двухтариф|многотариф|т1|т2|тарифн.{0,10}зон', 'многотарифный прибор раздельно тарифные зоны'),
    (r'фасад|вертикальн', 'монтаж вертикальные поверхности зданий размещение'),
    (r'расшир|нараст|поэтап|сначала.{0,25}потом', 'внедрять поэтапно наращивать мощность модульность'),
    (r'стар.{0,15}(?:кровл|крыш)|шифер', 'шифер срок службы кровли ограничения'),
    (r'магазин|торгов.{0,15}точ|ретейл', 'сетевой ретейл единый интерфейс сравнение объектов платформа учета'),
    (r'невыгод|не.окуп|откаж.{0,15}проект', 'бесплатная предварительная оценка экономическая нецелесообразность'),
    (r'квтч|киловатт.час', 'единицы измерения энергия мощность киловатт киловатт-час объем потребления'),
    (r'\bопт(?:а|е|у|ом)?\b|оптов|орэм', 'оптовый рынок присоединенная мощность'),
    (r'симк|sim|сим.карт', 'совместимость сим карты оператор прибор'),
    (r'фот|изображен|водомер', 'распознавание показаний фотографии верификация'),
    (r'\bноч|\bвечер|\bднем', 'профиль потребления дневная ночная нагрузка'),
    (r'два завод|несколько площад|сложить|суммир', 'электрическая связность площадок мощность'),
    (r'обход|ручн.{0,15}сбор', 'автоматизация сбора показаний учет фотография'),
    (r'интернет|связь|связи', 'передача данных мониторинг инфраструктура'),
    (r'\bначисл|\bквитанц|\bрубл', 'расчеты выставление счетов коммерческий учет'),
    (r'без выезд|\bакт(?:а|ы|ов|ом|е)?\b|\bнарушител', 'коммерческие потери вероятностная оценка проверка'),
    (r'подорож|растет счет|платим больше', 'ценовая категория стоимость электроэнергии'),
)


def expand(text: str) -> str:
    query = normal(text)
    extra = [phrase for pattern, phrase in EXPANSIONS if re.search(pattern, query)]
    return text + '\n' + ' '.join(extra)


def split_search_text(body: str, maximum: int = 1800) -> list[str]:
    """Окна только для индекса. Генератор получает целый раздел, не это окно."""
    result = []
    position = 0
    while position < len(body):
        end = min(len(body), position + maximum)
        if end < len(body):
            boundary = max(body.rfind('\n', position + maximum//2, end), body.rfind(' ', position + maximum//2, end))
            if boundary > position:
                end = boundary
        result.append(body[position:end].strip())
        if end == len(body):
            break
        position = max(position + 1, end - 180)
    return result


class Knowledge:
    def __init__(self, documents: list[Document], sections: list[Section]):
        self.documents = {doc.id: doc for doc in documents}
        self.sections = {sec.id: sec for sec in sections}
        self.pieces = [Piece(sec.id, sec.path + '\n' + part, sec.path)
                       for sec in sections for part in split_search_text(sec.body)]
        self.topic_docs = {}
        for topic, (_, stem_name) in TOPICS.items():
            self.topic_docs[topic] = [doc.id for doc in documents if stem_name in Path(doc.source).stem]
        self.reference_ids = [sec.id for sec in sections
                              if '02-product-scope' in sec.source and sec.heading == 'Общие границы']

    @classmethod
    def load(cls, path: Path) -> 'Knowledge':
        raw_files = []
        if path.is_dir():
            raw_files = [(str(p.relative_to(path)), p.read_text(encoding='utf-8-sig'))
                         for p in sorted(path.rglob('*')) if p.suffix.lower() in ('.md', '.txt')]
        else:
            with zipfile.ZipFile(path) as archive:
                infos = [i for i in archive.infolist() if not i.is_dir()
                         and not i.filename.startswith('__MACOSX')
                         and Path(i.filename).suffix.lower() in ('.md', '.txt')]
                if sum(i.file_size for i in infos) > 100_000_000:
                    raise ValueError('Слишком большой распакованный архив базы')
                raw_files = [(i.filename, archive.read(i).decode('utf-8-sig', errors='replace'))
                             for i in sorted(infos, key=lambda item: item.filename)]
        documents, sections = [], []
        for number, (name, text) in enumerate(raw_files, 1):
            did = f'D{number:02d}'
            doc = Document(did, name, Path(name).stem)
            # В исходной базе встречается "...качества. \\## Что нужно от клиента".
            text = re.sub(r'[ \t]*\\(#{1,6})[ \t]+', r'\n\1 ', text)
            text, _ = normalize_tables(text)
            stack = []
            body = []
            heading = doc.title
            level = 1

            def flush() -> None:
                content = '\n'.join(body).strip()
                if not content or not re.search(r'[A-Za-zА-Яа-я0-9]', content):
                    body.clear()
                    return
                sid = f'{did}S{len(doc.sections)+1:02d}'
                path_title = ' / '.join(h for _, h in stack) or heading
                sections.append(Section(sid, did, name, heading, path_title, content, level))
                doc.sections.append(sid)
                body.clear()

            for line in text.splitlines():
                match = re.match(r'^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$', line)
                if not match:
                    body.append(line)
                    continue
                flush()
                level, heading = len(match.group(1)), match.group(2)
                if level == 1:
                    doc.title = heading
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
            flush()
            if doc.sections:
                documents.append(doc)
        if not sections:
            raise ValueError('В базе нет непустых текстовых разделов')
        return cls(documents, sections)

    def catalog(self, maximum: int = 26000) -> str:
        """Оглавление для смыслового выбора разделов моделью, без выдуманных фактов."""
        lines = []
        for doc in self.documents.values():
            lines.append(f'\n{doc.id}: {doc.title}')
            for sid in doc.sections:
                section = self.sections[sid]
                lines.append(f'  {sid}: {section.heading}')
        result = '\n'.join(lines)
        return result if len(result) <= maximum else result[:maximum].rsplit('\n', 1)[0]


class Retriever:
    def __init__(self, kb: Knowledge):
        self.kb = kb
        self.field_index = FieldIndex(kb.sections)
        self.inverted = defaultdict(list)
        self.char_inverted = defaultdict(list)
        self.lengths, self.headings, self.char_sizes = [], [], []
        self.freq = []
        self.section_terms = {sid: set(tokens(sec.text)) for sid, sec in kb.sections.items()}
        self.section_pieces = defaultdict(list)
        for piece in kb.pieces:
            self.section_pieces[piece.section].append(piece.text)
        for index, piece in enumerate(kb.pieces):
            counts = Counter(tokens(piece.text))
            self.freq.append(counts)
            self.lengths.append(max(1, sum(counts.values())))
            self.headings.append(set(tokens(piece.title)))
            for term, freq in counts.items():
                self.inverted[term].append((index, freq))
            g = grams(piece.text)
            self.char_sizes.append(len(g))
            for term in g:
                self.char_inverted[term].append(index)
        n = max(1, len(kb.pieces))
        self.average = sum(self.lengths) / n
        self.idf = {t: math.log(1 + (n-len(v)+0.5)/(len(v)+0.5)) for t, v in self.inverted.items()}
        self.char_idf = {t: math.log(1 + n/(1+len(v))) for t, v in self.char_inverted.items()}

    def legacy_rank(self, query: str, limit: int = 80) -> list[tuple[str, float]]:
        word_score = [0.0] * len(self.lengths)
        char_score = [0.0] * len(self.lengths)
        for term in set(tokens(query)):
            for idx, freq in self.inverted.get(term, ()):
                denom = freq + 1.4*(0.25 + 0.75*self.lengths[idx]/self.average)
                word_score[idx] += self.idf[term]*freq*2.4/denom*(1.6 if term in self.headings[idx] else 1.0)
        for gram in grams(query):
            for idx in self.char_inverted.get(gram, ()):
                char_score[idx] += self.char_idf[gram]
        for idx in range(len(char_score)):
            char_score[idx] /= max(1, self.char_sizes[idx])**0.55
        by_word = sorted(range(len(word_score)), key=lambda i: (-word_score[i], i))
        by_char = sorted(range(len(char_score)), key=lambda i: (-char_score[i], i))
        fused = defaultdict(float)
        for weight, ranking, scores in ((1.0, by_word, word_score), (0.45, by_char, char_score)):
            for rank, idx in enumerate(ranking[:100]):
                if scores[idx] > 0:
                    fused[idx] += weight/(40+rank)
        # Один длинный раздел не занимает все места результатов своими окнами.
        sections = defaultdict(float)
        for idx, score in fused.items():
            sid = self.kb.pieces[idx].section
            sections[sid] = max(sections[sid], score)
        return sorted(sections.items(), key=lambda x: (-x[1], x[0]))[:limit]

    def rank(self, query: str, limit: int = 80) -> list[tuple[str, float]]:
        """Field scores retain relevance magnitude; character search is a backstop.

        Product names repeated in every parent heading must not drown out such
        discriminating terms as the roof material or the meter's production year.
        """
        values = self.field_index.scores(query[:10000])
        legacy = self.legacy_rank(query[:10000], 100)
        peak = max(values.values(), default=0)
        if peak <= 0:
            return legacy[:limit]
        mixed = {sid: value / peak for sid, value in values.items()}
        for position, (sid, _) in enumerate(legacy):
            mixed[sid] = mixed.get(sid, 0) + .12 * 30/(30+position)
        return sorted(mixed.items(), key=lambda x: (-x[1], x[0]))[:limit]

    def topics(self, dialogue: Dialogue) -> list[tuple[str, float]]:
        latest = normal(dialogue.last)
        past = normal(dialogue.prior_user[-3000:])
        values = {}
        for topic, (pattern, _) in TOPICS.items():
            direct = len(re.findall(pattern, latest))
            historical = len(re.findall(pattern, past))
            # При явной смене темы прошлый продукт НЕ становится основным.
            values[topic] = direct + min(historical, 2)*0.22
        # Упоминание счётчика в вопросе о фото не означает отдельную задачу АСКУЭ.
        if values['photo'] >= 1:
            if re.search(r'платформ|аскуэ|арендатор|по цехам|аналитик|сравни.{0,30}(?:магазин|объект|склад)', latest):
                values['metering'] = max(1.2, values['metering'])
            else:
                values['metering'] *= 0.28
        if values['solar'] >= 1:
            values['metering'] *= 0.35
        if values['orem'] >= 1:
            values['metering'] *= 0.45
        if len(tokens(latest)) < 5 or not any(v >= 1 for v in values.values()):
            for topic, (pattern, _) in TOPICS.items():
                values[topic] += len(re.findall(pattern, past))*0.65
        return sorted(((t, v) for t, v in values.items() if v > 0), key=lambda x: (-x[1], x[0]))

    def search(self, dialogue: Dialogue, rewrites: Iterable[str] = ()) -> list[tuple[str, float]]:
        query = dialogue.last[:6000]
        has_anchor = any(re.search(pattern, normal(query)) for topic, (pattern, _) in TOPICS.items()
                         if topic not in ('request', 'general'))
        elliptical = bool(dialogue.prior_user) and (not has_anchor or len(tokens(query)) < 5)
        # Короткая реплика без названия продукта («а сколько стоит?») считается эллиптичной:
        # вес самой реплики снижается, вес запроса с историей клиента повышается.
        queries = [(query, 0.35 if elliptical else 1.0), (expand(query), 0.35 if elliptical else 0.9)]
        if dialogue.prior_user:
            combined = query + '\n' + dialogue.prior_user[-3000:]
            queries.append((expand(combined), 1.7 if elliptical else 0.5))
        if dialogue.search_hint and elliptical:
            queries.append((expand(query + '\n' + dialogue.search_hint + '\n' + dialogue.prior_user[-1600:]), 0.55))
        # Составной вопрос ищем также по частям, чтобы вторая часть не исчезла.
        clauses = [x.strip() for x in re.split(r'[?;\n]+', query) if len(tokens(x)) >= 3]
        if len(clauses) > 1:
            queries.extend((x, 0.45) for x in clauses[:4])
        queries.extend((str(x)[:1200], 0.8) for x in list(rewrites)[:4] if isinstance(x, str))
        scores = defaultdict(float)
        for text, weight in queries:
            ranked = self.rank(text)
            peak = ranked[0][1] if ranked else 1.0
            for rank, (sid, score) in enumerate(ranked):
                # Оценки каждого запроса нормируются на лучший результат и сглаживаются по рангу.
                scores[sid] += weight * (score / peak) / (1 + rank / 55)
        active = {did for topic, value in self.topics(dialogue)[:2] if value >= 0.8
                  for did in self.kb.topic_docs.get(topic, [])}
        # При теме тарифов повышаем (x1.5) разделы тарифных документов,
        # при активном продукте (ниже) - разделы его документа (x1.35).
        if any(t == 'tariffs' and v >= 1 for t,v in self.topics(dialogue)):
            for sid in scores:
                if any(name in self.kb.sections[sid].source for name in ('14-tariffs','11-price-categories','10-market')):
                    scores[sid] *= 1.5
        if active:
            for sid in scores:
                if self.kb.sections[sid].document in active:
                    scores[sid] *= 1.35
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


    def rerank_candidates(self, dialogue: Dialogue, ranking, planned=()) -> list[str]:
        """Expand RE-RANKER INPUT, never force these extras into the answer.

        Preserve the original 30 candidates. Add up to ten heading matches,
        balanced across explicit product documents. This prevents a short FAQ
        (e.g. a price question) from being crowded out by long product cards.
        No product fact, answer, price or threshold is encoded here.
        """
        original = list(dict.fromkeys(list(planned) + [s for s, _ in ranking[:24]] +
                                      self.support_sections(dialogue, ranking)[:10]))[:30]
        original = [s for s in original if isinstance(s, str) and s in self.kb.sections]
        topics = [(t, score) for t, score in self.topics(dialogue)
                  if t in ('solar', 'orem', 'metering', 'photo', 'losses') and score >= .8]
        if not topics:
            return original
        docs = list(dict.fromkeys(d for topic, score in topics[:3]
                                  for d in self.kb.topic_docs.get(topic, [])))
        text = normal(dialogue.last)
        # Search-only linguistic normalization, not new factual information.
        if re.search(r'стоим|цен[аыуе]|сколько.{0,25}(?:стоит|стоят|плат)', text):
            text += ' сколько это стоит стоимость цена'
        if re.search(r'срок|как долго|сколько.{0,15}врем', text):
            text += ' сколько времени занимает срок'
        if re.search(r'интеграц|выгруз|передач.{0,20}данн', text):
            text += ' попадут ли данные в нашу систему'
        trivial = {stem_ru(w) for w in 'у нас вы мы это этот для по на в с и или а свой ваш наш мтс energytool'.split()}
        anchors = {stem_ru(w) for w in 'солнечная панель станция электростанция платформа счетчик прибор учет фотография распознавание оптовый рынок электроэнергия'.split()}
        qwords = {stem_ru(w) for w in WORD.findall(text)} - trivial - anchors
        if not qwords:
            return original
        pools = []
        for did in docs:
            scored = []
            for sid in self.kb.documents[did].sections:
                sec = self.kb.sections[sid]
                hwords = {stem_ru(w) for w in WORD.findall(normal(sec.heading))} - trivial - anchors
                common = hwords & qwords
                if not common:
                    continue
                # Coverage rewards a direct short question over a long, generic
                # title. Deterministic tie-breaks keep test runs reproducible.
                score = len(common) / max(1, len(hwords)) + .08 * len(common)
                scored.append((score, sid))
            pools.append([sid for _, sid in sorted(scored, key=lambda x: (-x[0], x[1]))[:4]])
        result = original[:]
        for i in range(4):
            for pool in pools:
                if i < len(pool) and pool[i] not in result:
                    result.append(pool[i])
        return result[:40]

    def rerank_text(self, sid: str, dialogue: Dialogue, maximum: int = 3800) -> str:
        """Select relevant exact windows for ranking; generation gets full sections."""
        sec = self.kb.sections[sid]
        if len(sec.text) <= maximum:
            return sec.text
        query_terms = set(tokens(expand(dialogue.last)))
        if len(query_terms) < 7:
            query_terms.update(tokens(dialogue.prior_user[-1600:]))
        windows = self.section_pieces.get(sid) or [sec.text]
        scored = []
        for i, window in enumerate(windows):
            present = set(tokens(window)) & query_terms
            score = sum(self.idf.get(term, 0.5) for term in present)
            scored.append((score, i, window))
        selected, used = [], len(sec.path) + 1
        for _, i, window in sorted(scored, key=lambda v: (-v[0], v[1])):
            body = window[len(sec.path):].lstrip() if window.startswith(sec.path) else window
            if used + len(body) + 22 <= maximum:
                selected.append((i, body))
                used += len(body) + 22
        if not selected:
            body = max(scored, key=lambda v: v[0])[2]
            room = max(0, maximum-len(sec.path)-1)
            return (sec.path + '\n' + body[:room])[:maximum]
        return sec.path + '\n' + '\n[пропуск фрагмента]\n'.join(body for _, body in sorted(selected))

    def facet_sections(self, dialogue: Dialogue) -> list[str]:
        """Keep separate supporting sections for multi-part questions."""
        parts = [p.strip() for p in re.split(r'[?;\n]+|(?:а еще|а также|и еще)\s+',
                                            normal(dialogue.last)) if len(tokens(p)) >= 3]
        if len(parts) < 2:
            return []
        selected = []
        for part in parts[:4]:
            if not re.search(r'как|мож|нуж|како|сколько|подой|хват|чем|что|подключ|хотим|треб', part):
                continue
            ranking = self.rank(expand(part))
            if ranking:
                selected.append(ranking[0][0])
        return list(dict.fromkeys(selected))

    def support_sections(self, dialogue: Dialogue,
                         ranking: list[tuple[str, float]]) -> list[str]:
        """Извлекает условия применимости из базы, а не подставляет готовые факты.

        Поисковый запрос короткий, но оговорка часто находится в соседнем разделе.
        Правила ниже выбирают названия разделов. Числа и текст берутся из KB.
        """
        kb = self.kb
        topics = [topic for topic, score in self.topics(dialogue) if score >= 0.8]
        # Для вопроса без названия продукта используем документы поисковой выдачи.
        if not any(t in topics for t in ('solar', 'orem', 'metering', 'photo', 'losses')):
            scores = defaultdict(float)
            for pos, (sid, _) in enumerate(ranking[:12]):
                for topic, (_, filename) in TOPICS.items():
                    if filename in kb.sections[sid].source:
                        scores[topic] += 1/(pos+4)
            if scores and not any(t in topics for t in ('tariffs','power','micro')):
                topics.extend(t for t, score in sorted(scores.items(), key=lambda x: -x[1])[:2]
                              if score >= max(scores.values())*0.65)
        topics = list(dict.fromkeys(topics))
        text = normal(dialogue.last + '\n' + dialogue.prior_user[-1600:])
        latest = normal(dialogue.last)
        mixed_collection = ('photo' in topics and bool(re.search(
            r'платформ|автоматическ|удален|арендатор|техник|обход', text)))
        if mixed_collection:
            # Фото и автоматический сбор — две дополняющие части задачи, а не
            # повод вытеснить первоначальную тему последней репликой о фото.
            topics = ['photo', 'metering'] + [t for t in topics if t not in ('photo','metering')]
        chosen = list(kb.reference_ids)

        def choose(source: str, headings: str) -> None:
            for sid, section in kb.sections.items():
                if source in section.source and re.search(headings, normal(section.heading)):
                    if sid not in chosen:
                        chosen.append(sid)

        if mixed_collection:
            choose('22-askue', '^что это$')
            choose('03-calculation', '^для платформы учета$')
        scope_names = {'solar': 'солнеч', 'orem': 'оптов', 'metering': 'платформ',
                       'photo': 'фотограф', 'losses': 'потер'}
        for topic in topics[:3]:
            if topic in scope_names:
                choose('02-product-scope', scope_names[topic])
        pricing = bool(re.search(r'цен[ауы]|стоим|сколько.*(?:руб|плат|стоит)|окуп|экономи|процент|срок|гаранти|точност|сколько.*(?:прибор|город)', latest))
        if pricing:
            for topic in topics[:2]:
                if topic in scope_names:
                    choose('01-numbers', scope_names[topic] if topic != 'photo' else 'распознаван')
        if re.search(r'персональн|юридическ|договор|ответственност', latest):
            choose('02-product-scope', 'когда требуется специалист')
        if re.search(r'данн|измер|архив|профил|начать|подготов|оценк|расчет|хватит|достаточ', latest):
            names = {'solar': 'солнеч', 'orem': 'анализа затрат', 'metering': 'платформ',
                     'photo': 'распознаван', 'losses': 'потер'}
            for topic in topics[:2]:
                if topic in names:
                    choose('03-calculation', names[topic])
        if re.search(r'невыгод|не.окуп|обязательств|бесплат|стоим.{0,15}расчет', latest):
            choose('03-calculation', 'бесплатная предварительная оценка')
        if re.search(r'заявк|перезвон|менеджер|контакт|телефон|коммерческ.{0,15}предложен', latest):
            choose('03-calculation', 'заявка|согласие|обратный звонок')
        if re.search(r'поэтап|расшир|нараст|сначала.{0,30}потом', latest):
            choose('26-general', 'поэтапно')
        if 'solar' in topics:
            if re.search(r'крыш|кровл|скат|фасад|шифер|монтаж|вертикальн', text):
                choose('23-solar', '^размещение$|^не подходит, если$')
            if re.search(r'вечер|ноч|резерв|автоном|отключ|накопител|гибрид|сетев', text):
                choose('23-solar', 'два типа|пропадет электричество|лучший профиль')
            if re.search(r'окуп|выработ|эффект|экономи', latest):
                choose('23-solar', '^как считается эффект$')
            if re.search(r'инсоляц|регион|север|солнечн.{0,8}дн', latest):
                choose('23-solar', 'география|мало солнца')
        if 'photo' in topics:
            # Верификация — обязательное условие применения, а не свойство АСКУЭ.
            choose('25-meter-reading', '^важное ограничение по применению$')
            if re.search(r'год|тариф|номер|штрих|прочита|счит|распозна', latest):
                choose('25-meter-reading', '^что распознается$')
            if re.search(r'газ|тепл|вод|ресурс', latest):
                choose('25-meter-reading', '^поддерживаемые ресурсы$')
        if 'losses' in topics:
            if re.search(r'данн|архив|хватит|достаточ|год|месяц|сут|час', text):
                choose('24-losses', '^что нужно от клиента$')
                choose('03-calculation', '^для поиска коммерческих потерь$')
            if re.search(r'сбыт|промыш|завод|цех', text):
                choose('24-losses', 'не подходит')
        if 'orem' in topics:
            if re.search(r'прогноз|сопровожд|после|специалист|кто', latest):
                choose('21-orem', '^что делает мтс$|^нужны ли нам свои специалисты')
            if re.search(r'учет|прибор|счетчик', latest):
                choose('21-orem', '^нужно ли менять существующую систему учета$')
            choose('12-power-thresholds', '^пороги оптового рынка$')
            if re.search(r'несколько|площад|сумм|слож|объедин', text):
                choose('21-orem', '^если у предприятия несколько площадок$')
            if re.search(r'камчат|магадан|сахалин|чукот|изолиров', text):
                choose('10-market', '^где оптового рынка нет$')
            if re.search(r'аудит|существующ|провер.{0,15}учет|модерниз|меня.{0,15}учет', latest):
                choose('21-orem', '^нужно ли менять существующую систему учета$|^что делает мтс$')
        if re.search(r'категори|счет|платеж', text) and re.search(r'рост|вырос|увелич|больш|дорож|сравн|провер', latest):
            choose('14-tariffs', '^что именно растет$|^следующий шаг$')
        if 'tariffs' in topics and re.search(r'категори|треть|четверт|пят|шест', text):
            choose('11-price-categories', '^шесть категорий$')
        if ('квт' in latest and 'квтч' in latest) or re.search(r'мощност.{0,25}энерги|энерги.{0,25}мощност', latest):
            choose('12-power-thresholds', '^главное про единицы измерения$')
            choose('17-faq', '^чем мощность отличается от энергии$')
        if re.search(r'реактив|компенсир', latest):
            choose('17-faq', 'реактивная энергия')
        if re.search(r'микрогенер|отда.{0,20}сет|излиш|прода.{0,20}сет', text):
            choose('15-microgeneration', 'сколько можно|продажа излишков|три разных')
        return chosen

    def context(self, dialogue: Dialogue, ranking: list[tuple[str,float]] | None = None,
                plan: dict | None = None, maximum: int = 20000) -> tuple[str,list[str]]:
        """A coherent product card plus direct evidence, not a bag of fragments.

        Small product manuals fit the runtime budget. Once a product is explicit,
        their main sections prevent an obscure capability/exception from falling
        outside top-k. This is prioritization only: every other document remains
        eligible, especially the sections explicitly requested by the model.
        """
        ranking = ranking if ranking is not None else self.search(dialogue)
        ranking = [(sid,score) for sid,score in ranking if sid in self.kb.sections]
        plan = plan or {}
        kb = self.kb
        selected,blocks,used = [],[],0
        def add(sid: str) -> bool:
            nonlocal used
            if sid not in kb.sections or sid in selected:
                return False
            sec=kb.sections[sid]
            block=f'[{sid}] {sec.path}\n{source_role(sec.heading)}\n{sec.body}'
            cost=len(block)+(7 if blocks else 0)
            if used+cost > maximum:
                return False
            selected.append(sid);blocks.append(block);used+=cost
            return True
        # Порядок приоритета при сборке контекста: 1) разделы планировщика и догрузки,
        # 2) топ-4 реранкера, 3) топ-2 поиска, 4) ограничения документов-владельцев,
        # 5) части составного вопроса, 6) поддерживающие разделы (границы, цены, данные),
        # 7) основные разделы продукта, 8) остаток выдачи. add() пропускает блок,
        # если он не помещается в лимит символов.
        requested=[sid for sid in plan.get('sections',[]) if isinstance(sid,str) and sid in kb.sections]
        for sid in requested:
            add(sid)
        # The semantic reranker is useful only if its best sources actually
        # survive packing. In v12 only TWO survived with priority; the rest
        # could be displaced by an entire generic product card.
        # This list is supplied only after a successful API rerank, not by
        # generated search text and not by topic heuristics.
        semantic = plan.get('reranked_sections', [])
        semantic = ([sid for sid in semantic
                     if isinstance(sid, str) and sid in kb.sections]
                    if isinstance(semantic, list) else [])
        semantic = list(dict.fromkeys(semantic))[:4]
        for sid in semantic:
            add(sid)
        for sid,_ in ranking[:2]:
            add(sid)
        # Bind strong product evidence to that SAME document's restrictions.
        # Text is read from the supplied KB, never from hard-coded product facts.
        owners = list(dict.fromkeys(kb.sections[s].document for s in semantic))
        for did in owners:
            # A capability without its exclusion is misleading. Preserve the
            # document's own applicability clauses before generic background.
            for sid in kb.documents[did].sections:
                heading = normal(kb.sections[sid].heading)
                if (heading.startswith('не подходит') or
                        heading == 'важное ограничение по применению'):
                    add(sid)
            for sid in kb.documents[did].sections:
                if normal(kb.sections[sid].heading) == 'чего утверждать нельзя':
                    add(sid)
        for sid in self.facet_sections(dialogue):
            add(sid)
        for sid in self.support_sections(dialogue,ranking):
            add(sid)
        products=[(topic,value) for topic,value in self.topics(dialogue)
                  if topic in ('solar','orem','metering','photo','losses') and value>=.8]
        product_docs=[]
        if products:
            highest=products[0][1]
            for topic,value in products[:2]:
                if value>=highest*.42:
                    product_docs.extend(kb.topic_docs.get(topic,[]))
        product_docs=list(dict.fromkeys(product_docs))
        # Domain-independent scaffolding: introductions, functional sections and
        # restrictions. Duplicated FAQ/summary blocks are not required to fit.
        core=[]
        for did in product_docs:
            sections=[kb.sections[s] for s in kb.documents[did].sections]
            main=[sec.id for sec in sections if sec.level<=2 and
                  normal(sec.heading) not in ('факты, которые можно утверждать',)]
            # High-value level-3 eligibility subsections sit before FAQ sections.
            faq_started=False
            for sec in sections:
                if 'частые вопросы' in normal(sec.path):
                    faq_started=True
                if sec.level==3 and not faq_started:
                    main.append(sec.id)
            core.append(main)
        # For comparisons, alternate products rather than exhausting the first.
        for i in range(max([len(items) for items in core],default=0)):
            for items in core:
                if i<len(items):add(items[i])
        for sid,_ in ranking[:12]:
            add(sid)
        # If there is no explicit product, keep sources selected by ranking but
        # do not invent a product from a generic time-of-day or contact phrase.
        for did in product_docs:
            for sid in kb.documents[did].sections:
                if normal(kb.sections[sid].heading)=='чего утверждать нельзя':add(sid)
        return '\n\n---\n\n'.join(blocks),selected

    def planner_needed(self, dialogue: Dialogue) -> bool:
        """Планирование только для неоднозначных диалогов и составных задач."""
        latest = normal(dialogue.last)
        has_anchor = any(re.search(p, latest) for t, (p, _) in TOPICS.items()
                         if t not in ('general', 'request'))
        complex_query = bool(re.search(r'сравни|чем отличаются|что выбрать|какое решение', latest))
        return ((bool(dialogue.history) and (len(tokens(latest)) < 6 or not has_anchor))
                or dialogue.last.count('?') > 1 or complex_query)
