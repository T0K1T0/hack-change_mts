"""Small local lexical index. No downloaded language models or data files.

Russian inflections are reduced before BM25. Headings and bodies are indexed
separately so a repeated product title cannot overwhelm the actual question.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

_WORD = re.compile(r'[a-zа-я0-9]+', re.I)
_STOP = set('''а и в во на с со к ко по для за из от до о об у же ли не ни но или
то так это эти этой тот что как какой какие где когда при если можно нужно наш наша
ваше вы мы они его ее их мой быть есть был были будет уже бы да нет очень просто
либо стоит хочу хотим могу можете пожалуйста расскажите скажите подскажите привет
добрый день вопрос этом этого также мне нам вас вам такой такого вообще нас у нас
ваш наша наши которые который чтобы чем только еще тут там это все всего самого
нужен нужна нужно нужны ведь значит получается у меня должен должна должно
работает работать работа сделать делаете сам сама своих свой своей своими
рассматривать рассмотреть разобраться почему вообще возможности вариант решение
предлагать предлагают сначала сразу начать знать понять сказать
кто кому кого какая какое каких каким какие сколько надо может сможем смогу
нам никто ничего почему зачем чего чем какой сколько некоторым такое тоже прочее'''.split())

@lru_cache(maxsize=30000)
def stem_ru(word: str) -> str:
    """Light rule-based Russian inflection reducer; not semantic translation."""
    word = word.lower().replace('ё', 'е')
    if len(word) < 4 or not re.fullmatch('[а-я]+', word):
        return word
    first = re.search('[аеиоуыэюя]', word)
    if not first:
        return word
    pre, rv = word[:first.end()], word[first.end():]
    # Perfective forms, reflexive suffixes, adjectives/participles, verbs, nouns.
    old = rv
    rv = re.sub(r'(?:ившись|ывшись|ивши|ывши|ив|ыв)$', '', rv)
    if rv == old:
        rv = re.sub(r'(?<=[ая])(?:вшись|вши|в)$', '', rv)
    if rv == old:
        rv = re.sub(r'(?:ся|сь)$', '', rv)
        before_adj = rv
        rv = re.sub(r'(?:ими|ыми|его|ого|ему|ому|ее|ие|ые|ое|ей|ий|ый|ой|ем|им|ым|ом|их|ых|ую|юю|ая|яя|ою|ею)$', '', rv)
        if rv != before_adj:
            rv = re.sub(r'(?:ивш|ывш|ующ)$', '', rv)
            rv = re.sub(r'(?<=[ая])(?:ем|нн|вш|ющ|щ)$', '', rv)
        else:
            before_verb = rv
            rv = re.sub(r'(?:ейте|уйте|ила|ыла|ена|ите|или|ыли|ило|ыло|ено|ует|уют|ены|ить|ыть|ишь|ей|уй|ил|ыл|им|ым|ен|ят|ит|ыт|ую|ю)$', '', rv)
            if rv == before_verb:
                rv = re.sub(r'(?<=[ая])(?:ете|йте|ешь|нно|ла|на|ли|ло|но|ет|ны|ть|й|л|н)$', '', rv)
            if rv == before_verb:
                rv = re.sub(r'(?:иями|ями|ами|ией|иям|ием|иях|ев|ов|ие|ье|еи|ии|ей|ой|ий|ям|ем|ам|ом|ах|ях|ию|ью|ия|ья|а|е|и|й|о|у|ы|ь|ю|я)$', '', rv)
    rv = re.sub(r'и$', '', rv)
    rv = re.sub(r'(?:ейше|ейш)$', '', rv)
    rv = re.sub(r'нн$', 'н', rv)
    rv = re.sub(r'ь$', '', rv)
    result = pre + rv
    return result if len(result) >= 3 else word


_STOP_STEMS = {stem_ru(w) for w in _STOP}


def terms(text: str) -> list[str]:
    value = text.lower().replace('ё', 'е').replace('\u00a0', ' ')
    value = re.sub(r'квт[·⋅* /-]?ч\b', 'квтч', value)
    return [v for w in _WORD.findall(value) if len(w) > 1 and w not in _STOP
            for v in [stem_ru(w)] if v not in _STOP_STEMS]


class FieldIndex:
    """BM25 with a separate heading field and low-weight parent title field."""
    def __init__(self, sections):
        self.sections = sections
        self.body = {sid: Counter(terms(sec.body)) for sid, sec in sections.items()}
        self.heading = {sid: set(terms(sec.heading)) for sid, sec in sections.items()}
        self.parent = {sid: set(terms(sec.path)) - self.heading[sid] for sid, sec in sections.items()}
        self.lengths = {sid: max(1, sum(tf.values())) for sid, tf in self.body.items()}
        self.average = sum(self.lengths.values()) / max(1, len(sections))
        df = Counter()
        for sid in sections:
            df.update(set(self.body[sid]) | self.heading[sid] | self.parent[sid])
        n = len(sections)
        self.idf = {t: math.log(1 + (n - d + .5) / (d + .5)) for t, d in df.items()}
        self.postings = defaultdict(set)
        for sid in sections:
            for t in set(self.body[sid]) | self.heading[sid] | self.parent[sid]:
                self.postings[t].add(sid)

    def scores(self, query: str) -> dict[str, float]:
        qt = set(terms(query))
        # Once a product has been named, "solar/panels/station" are routing
        # terms. "slate/roof/east/west" discriminate its actual FAQ.
        anchors = set(terms('солнечная панель станция электростанция платформа '
                            'счетчик прибор учет фотография распознавание '
                            'оптовый рынок электроэнергия energytool мтс'))
        informative = qt - anchors
        # Если в запросе есть значимые слова помимо названий продуктов, вес общих
        # «якорных» слов (солнечная, платформа, прибор...) снижается до 0.22.
        reduce_anchors = len(informative) >= 2
        out = defaultdict(float)
        for t in qt:
            variants = {t: 1.}
            if len(t) >= 4:
                for v in self.postings:
                    short, long = (t, v) if len(t) <= len(v) else (v, t)
                    if (len(short) >= 4 and long.startswith(short) and
                        len(short) / len(long) >= .78 and
                        long[len(short):] in ('н', 'ск', 'ов', 'ев', 'ш')):
                        variants[v] = .85 if v != t else 1.
            best = {}
            for variant, similarity in variants.items():
                for sid in self.postings.get(variant, ()):
                    f = self.body[sid].get(variant, 0)
                    norm = f + 1.35 * (.35 + .65 * self.lengths[sid] / self.average)
                    body = 2.35 * f / norm if f else 0
                    head = 1.4 if variant in self.heading[sid] else 0
                    parent = .10 if variant in self.parent[sid] else 0
                    weight = .22 if reduce_anchors and t in anchors else 1.
                    # Итог по слову: idf * (BM25 по телу + 1.4 за совпадение в заголовке +
                    # 0.1 за совпадение в родительском заголовке); similarity < 1 для близких словоформ.
                    score = weight * similarity * self.idf[variant] * (body + head + parent)
                    best[sid] = max(best.get(sid, 0), score)
            for sid, value in best.items():
                out[sid] += value
        return dict(out)

    def rank(self, query: str, limit: int = 80):
        values = self.scores(query)
        return sorted(values.items(), key=lambda p: (-p[1], p[0]))[:limit]
