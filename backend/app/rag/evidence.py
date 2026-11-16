"""Lossless layout recovery for organizer Markdown tables; no product facts here.

The source is Pandoc's fixed-width/multiline Markdown. A number on one visual
row and its qualifier on another must remain a single fact. Unknown layouts
are left untouched. No language model is used to rewrite source facts.
"""
from __future__ import annotations
import re
from collections import Counter

MULTI = re.compile(r'^\s*-{3,}(?:[ \t]+-{3,})+[ \t]*$')
BORDER = re.compile(r'^\s*-{12,}\s*$')
TOKEN = re.compile(r'\w+|[^\w\s]', re.UNICODE)


def _cells(lines: list[str], starts: list[int]) -> list[str] | None:
    result = [[] for _ in starts]
    for line in lines:
        # A malformed layout may split a word. Do not invent a reconstruction.
        for start in starts[1:]:
            if len(line) > start and not line[start-1].isspace() and not line[start].isspace():
                return None
        for i, start in enumerate(starts):
            stop = starts[i+1] if i+1 < len(starts) else len(line)
            part = line[start:stop].strip()
            if part:
                result[i].append(part)
    cells = [' '.join(x) for x in result]
    # This check catches dropped text before the first column, among other errors.
    if Counter(TOKEN.findall('\n'.join(lines))) != Counter(TOKEN.findall(' '.join(cells))):
        return None
    return cells


def normalize_tables(text: str) -> tuple[str, list[dict]]:
    """Return normalized text and provenance, retaining every body-cell token.

    Each reconstructed record repeats its column names. Tables with uncertain
    boundaries, mixed layouts or split tokens retain their exact original text.
    """
    lines = text.splitlines()
    replacements = []
    audit = []
    consumed = -1
    for i, line in enumerate(lines):
        if i <= consumed or not MULTI.fullmatch(line):
            continue
        starts = [m.start() for m in re.finditer(r'-{3,}', line)]
        a = i-1
        while a >= 0 and lines[a].strip() and not BORDER.fullmatch(lines[a]) and not lines[a].lstrip().startswith('#'):
            a -= 1
        bordered = a >= 0 and bool(BORDER.fullmatch(lines[a]))
        hstart = a+1
        if hstart >= i:
            continue
        headers = _cells(lines[hstart:i], starts)
        if headers is None:
            continue
        end = i+1
        if bordered:
            while end < len(lines) and not BORDER.fullmatch(lines[end]):
                if lines[end].lstrip().startswith('#'):
                    break
                end += 1
            if end >= len(lines) or not BORDER.fullmatch(lines[end]):
                continue
            data_lines = lines[i+1:end]
            groups = []
            group = []
            for value in data_lines + ['']:
                if value.strip():
                    group.append(value)
                elif group:
                    groups.append(group); group = []
            stop = end+1
            begin = a
        else:
            while end < len(lines) and lines[end].strip() and not lines[end].lstrip().startswith('#'):
                end += 1
            groups = [[x] for x in lines[i+1:end] if x.strip()]
            stop = end
            begin = hstart
        if not groups:
            continue
        rows = [_cells(g, starts) for g in groups]
        if any(row is None for row in rows):
            continue
        body_tokens = Counter(TOKEN.findall('\n'.join(x for g in groups for x in g)))
        cell_tokens = Counter(TOKEN.findall(' '.join(x for row in rows for x in row)))
        if body_tokens != cell_tokens:
            continue
        labels = [h if h else ('Параметр' if j == 0 else f'Столбец {j+1}') for j,h in enumerate(headers)]
        rendered = ['Таблица: ' + ' | '.join(labels)]
        for row in rows:
            rendered.append('\n' + '; '.join(f'{label}: {cell or "—"}' for label,cell in zip(labels,row)) + '.')
        new = '\n'.join(rendered)
        old = '\n'.join(lines[begin:stop])
        replacements.append((begin,stop,new))
        audit.append({'line':begin+1, 'rows':len(rows), 'columns':len(starts),
                      'source_chars':len(old), 'normalized_chars':len(new),
                      'body_tokens_preserved':True,
                      'headers':headers, 'cells':rows})
        consumed = stop-1
    for start,stop,value in reversed(replacements):
        lines[start:stop] = [value]
    return '\n'.join(lines), audit


def source_role(heading: str) -> str:
    h = heading.lower().replace('ё','е')
    if 'чего утверждать нельзя' in h or 'что мы не делаем' in h:
        return 'ОГРАНИЧЕНИЯ: пункты о недопустимых обещаниях — это запреты, не свойства продукта'
    if 'оговор' in h or 'область применения' in h or 'не подходит' in h:
        return 'УСЛОВИЯ ПРИМЕНИМОСТИ'
    return 'МАТЕРИАЛ БАЗЫ'
