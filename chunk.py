from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
import sys
import unicodedata
import pdfplumber
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

import config

MIN_WORDS = 100
TARGET_WORDS = 400
MAX_WORDS = 500
OVERLAP_WORDS = 50
FALLBACK_MIN_HEADINGS = 3

EXCLUDED_SECTION= (
    "references", "bibliography", "acknowledgements", "acknowledgments", "acknowledgement",
    "acknowledgment", "author contributions", "declaration of conflicting interests",
    "contents", "table of contents",
)

UNNUMBERED_HEADINGS = {
    "abstract", "references", "bibliography", "acknowledgements", "acknowledgments",
    "acknowledgement", "acknowledgment", "conclusion", "conclusions", "limitations",
    "contents", "table of contents", "appendix", "ethics statement", "ethical considerations",
    "author contributions", "declaration of conflicting interests", "discussion",
    "introduction", "related work", "related works", "background",
}

BOLD_MARKERS = ("bold", "semibold", "demi", "medi", "bx", "cmbx", "sfbx", "-bd", "heavy", "black")

NUMBERED_RE = re.compile(r"^(?P<num>(?:\d{1,2}|[A-H])(?:\.\d{1,2}){0,3})\.?\s+(?P<title>[A-Z][A-Za-z][^\n]*)$")

@dataclass
class Line:
    page: int
    text: str
    top: float
    bottom: float
    x0: float
    x1: float
    size: float
    bold_frac: float
    font: str
    n_gaps: int = 0

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass
class Heading:
    number: str
    title: str
    depth: int
    line_index: int
    page: int
    last_line: int = -1


@dataclass
class Section:
    heading: Optional[Heading]
    lines: list = field(default_factory=list)
    paragraphs: list = field(default_factory=list)


def paper_id_from_name(name: str) -> tuple[str, str]:
    m = re.search(r"(\d{4})[._](\d{4,5})(v\d+)?", name)
    if m:
        return f"{m.group(1)}.{m.group(2)}", (m.group(3) or "")
    m = re.search(r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", name)
    if m:
        return m.group(1).replace("/", "_"), (m.group(2) or "")
    return Path(name).stem, ""


def is_bold(fontname: str) -> bool:
    f = fontname.lower().split("+")[-1]
    return any(mk in f for mk in BOLD_MARKERS)


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\(cid:\d+\)", "", s)
    s = s.replace("­", "")
    s = re.sub(r"[ -​  ]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def word_count(s: str) -> int:
    return len(s.split())


def norm_heading_name(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def is_excluded_name(title: str) -> bool:
    n = norm_heading_name(title)
    return any(n == e or n.startswith(e + " ") for e in EXCLUDED_SECTION)


#cleaned lines with font information

def _extract_page_lines(page, pageno: int) -> list[Line]:
    #drop rotated text
    page = page.filter(lambda o: o.get("object_type") != "char" or o.get("upright", True))
    words = page.extract_words(
        x_tolerance=1.5, y_tolerance=2, use_text_flow=True,
        extra_attrs=["fontname", "size"], keep_blank_chars=False,
    )
    lines: list[dict] = []
    cur = None
    for w in words:
        if cur is not None:
            same_row = (w["bottom"] <= cur["bottom"] + 2.5
                        and w["bottom"] >= cur["top"] + 0.5 * (cur["bottom"] - cur["top"])
                        and w["x0"] >= cur["x1"] - 3)
        else:
            same_row = False
        if same_row:
            gap = w["x0"] - cur["x1"]
            cur["parts"].append((w["text"], gap, w["fontname"], w["size"]))
            cur["x1"] = w["x1"]
            cur["top"] = min(cur["top"], w["top"])
            cur["bottom"] = max(cur["bottom"], w["bottom"])
        else:
            if cur is not None:
                lines.append(cur)
            cur = dict(parts=[(w["text"], 0.0, w["fontname"], w["size"])],
                       top=w["top"], bottom=w["bottom"], x0=w["x0"], x1=w["x1"])
    if cur is not None:
        lines.append(cur)

    out: list[Line] = []
    for l in lines:
        text = " ".join(p[0] for p in l["parts"])
        weights = collections.Counter()
        bold_chars = 0
        total = 0
        for t, _, fn, sz in l["parts"]:
            n = len(t)
            weights[(fn, round(sz, 1))] += n
            total += n
            if is_bold(fn):
                bold_chars += n
        (fn, sz), _ = weights.most_common(1)[0]
        n_gaps = sum(1 for p in l["parts"][1:] if p[1] > 0.9 * sz)
        out.append(Line(page=pageno, text=text, top=l["top"], bottom=l["bottom"],
                        x0=l["x0"], x1=l["x1"], size=sz,
                        bold_frac=bold_chars / max(total, 1), font=fn, n_gaps=n_gaps))
    return out


def extract_lines(pdf_path: Path) -> tuple[list[Line], dict]:
    all_lines: list[Line] = []
    char_sizes = collections.Counter()
    page_heights = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        n_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages, 1):
            page_heights[i] = float(page.height)
            for ch in page.chars:
                if ch.get("upright", True):
                    char_sizes[round(ch["size"], 1)] += 1
            all_lines.extend(_extract_page_lines(page, i))

    body_size = char_sizes.most_common(1)[0][0] if char_sizes else 10.0

    #remove running headers/footers
    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.lower()).strip()

    margin_counts = collections.Counter()
    for l in all_lines:
        H = page_heights[l.page]
        if l.top < 0.09 * H or l.bottom > 0.93 * H:
            margin_counts[norm(l.text)] += 1
    repeated = {t for t, c in margin_counts.items() if c >= max(3, 0.3 * n_pages)}

    kept: list[Line] = []
    dropped = collections.Counter()
    for l in all_lines:
        H = page_heights[l.page]
        in_margin = l.top < 0.09 * H or l.bottom > 0.93 * H
        t = clean_text(l.text)
        if not t:
            continue
        if in_margin and norm(l.text) in repeated:
            dropped["running_header_footer"] += 1
            continue
        if in_margin and re.fullmatch(r"[\divxlc\-–—\s]{1,6}", t.lower()):
            dropped["page_number"] += 1
            continue
        if l.size < 0.80 * body_size:
            dropped["small_font(figure/table/footnote)"] += 1
            continue
        l.text = t
        kept.append(l)

    diag = dict(n_pages=n_pages, body_size=body_size, dropped=dict(dropped))
    return kept, diag


#heading detection

def detect_heading(line: Line, body_size: float) -> Optional[Heading]:
    t = line.text
    if len(t) > 110 or line.bold_frac < 0.8 or line.size < 0.97 * body_size:
        return None
    m = NUMBERED_RE.match(t)
    if m:
        title = m.group("title").strip()
        if re.search(r"[.!?]\s+[A-Z][a-z]+ [a-z]+ [a-z]+", title):
            return None
        if re.search(r"\s\d{1,3}$", title) and not re.search(r"[A-Za-z]\s\d$", title[:-1] + "x"):
            return None
        alpha = sum(c.isalpha() for c in title) / max(len(title), 1)
        if alpha < 0.6:
            return None
        num = m.group("num")
        return Heading(number=num, title=title.rstrip(".:"), depth=num.count(".") + 1,
                       line_index=-1, page=line.page)
    if re.search(r"[.!?]\s+[A-Z][a-z]+ [a-z]+ [a-z]+", t):
        return None
    name = norm_heading_name(t)
    if name in UNNUMBERED_HEADINGS and len(t) < 60:
        return Heading(number="", title=t.rstrip(".:"), depth=1, line_index=-1, page=line.page)
    return None


def find_headings(lines: list[Line], body_size: float) -> list[Heading]:
    heads = []
    skip_until = -1
    for i, l in enumerate(lines):
        if i <= skip_until:
            continue
        h = detect_heading(l, body_size)
        if not h:
            continue
        h.line_index = i
        h.last_line = i
        j = i + 1
        while j < len(lines):
            n, p = lines[j], lines[j - 1]
            if (n.page == l.page and n.bold_frac >= 0.8 and abs(n.size - l.size) < 0.3
                    and 0 <= n.top - p.bottom < 0.6 * max(p.height, 1)
                    and len(n.text) < 80 and not detect_heading(n, body_size)
                    and not h.title.endswith((".", ":", "?"))):
                h.title = (h.title + " " + n.text).strip().rstrip(".:")
                h.last_line = j
                j += 1
            else:
                break
        skip_until = h.last_line
        heads.append(h)
    return heads



#drop table/equation residue)

CAPTION_RE = re.compile(r"^(Figure|Fig\.|Table)\s+[A-Z]?\d+[:.]", re.I)


def _looks_like_table_or_equation(text: str, n_gaps: int = 0) -> bool:
    toks = text.split()
    if not toks:
        return True
    if CAPTION_RE.match(text):
        return False                                        #keep captions
    if n_gaps >= 2 or (n_gaps == 1 and len(toks) <= 12 and
                       sum(bool(re.search(r"\d", t)) for t in toks) / len(toks) >= 0.3):
        return True                                         #tabular row
    num = sum(bool(re.fullmatch(r"[-+±~≈<>(]*\d[\d.,%)*†‡]*", t)) for t in toks)
    if len(toks) >= 5 and num / len(toks) > 0.55:
        return True                                         # table row
    alpha = sum(c.isalpha() for c in text) / max(len(text), 1)
    if len(toks) <= 14 and alpha < 0.35:
        return True                                         #equation marker
    if re.fullmatch(r"\(?\d{1,3}\)?", text):
        return True                                         #equation number
    return False


def build_vocab(lines: Iterable[Line]) -> set[str]:
    vocab = set()
    for l in lines:
        for w in re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", l.text):
            vocab.add(w.lower())
    return vocab


def _tidy(text: str) -> str:
    #delete spacing for math
    text = re.sub(r"(?<=\d) \. (?=\d)", ".", text)
    text = re.sub(r"(?<=\d) \.(?=\d)", ".", text)
    text = re.sub(r"(?<=[−-]) (?=\d)", "", text)
    text = re.sub(r"\( ", "(", text)
    text = re.sub(r" \)", ")", text)
    text = re.sub(r"\s+([,;])(?=\s|$)", r"\1", text)
    text = re.sub(r"(?<=\d)\s+%", "%", text)
    text = re.sub(r"(?<=\w)- ([a-z])\b(?=\s)", r"-\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _ends_sentence(t: str) -> bool:
    return t.rstrip().endswith((".", "!", "?", ":", ")")) and not t.rstrip().endswith(("et al.", "e.g.", "i.e."))


def lines_to_paragraphs(lines: list[Line], vocab: set[str]) -> list[tuple[str, int]]:
    #group lines into paragraphs
    if not lines:
        return []
    heights = [l.height for l in lines if l.height > 0]
    lh = statistics.median(heights) if heights else 11.0
    body_size = statistics.median(l.size for l in lines)
    paras: list[list[Line]] = []
    prev: Optional[Line] = None
    for l in lines:
        if _looks_like_table_or_equation(l.text, l.n_gaps):
            continue
        new_para = prev is None or bool(CAPTION_RE.match(l.text))
        if not new_para and l.page == prev.page and abs(l.x0 - prev.x0) < 150:
            gap = l.top - prev.bottom
            indented = 4 <= l.x0 - prev.x0 <= 30 and _ends_sentence(prev.text)
            if gap > 0.9 * lh or indented or abs(l.size - prev.size) > 0.4:
                new_para = True
        if new_para or not paras:
            paras.append([l])
        else:
            paras[-1].append(l)
        prev = l

    built = []
    for p in paras:
        text = ""
        for l in p:
            t = l.text
            if not text:
                text = t
            elif text.endswith("-") and t[:1].islower() and re.search(r"[A-Za-z]-$", text):
                head = re.search(r"([A-Za-z]+)-$", text).group(1)
                m = re.match(r"[A-Za-z]+", t)
                tail = m.group(0) if m else ""
                if (head + "-" + tail).lower() in vocab and (head + tail).lower() not in vocab:
                    text = text + t
                else:
                    text = text[:-1] + t
            else:
                text = text + " " + t
        text = _tidy(text)
        if text:
            small = statistics.median(l.size for l in p) < 0.92 * body_size
            built.append((text, p[0].page, bool(CAPTION_RE.match(text)) or (small and len(text.split()) < 40)))

    #place captions after the paragraph they interrupted
    res: list[tuple[str, int]] = []
    pend: list[tuple[str, int]] = []
    for text, pg, cap in built:
        if cap:
            pend.append((text, pg))
            continue
        if pend and res and not _ends_sentence(res[-1][0]) and text[:1].islower():
            res[-1] = (res[-1][0] + " " + text, res[-1][1])
            res.extend(pend)
        else:
            res.extend(pend)
            res.append((text, pg))
        pend = []
    res.extend(pend)
    return res



#build sections
def _title_from_first_page(lines: list[Line]) -> tuple[str, int]:
    first = [l for l in lines if l.page == 1]
    if not first:
        return "", 0
    biggest = max(l.size for l in first)
    title_lines = []
    idx_end = 0
    for i, l in enumerate(lines):
        if l.page != 1:
            break
        if l.size >= 0.9 * biggest and l.size > 12:
            title_lines.append(l.text)
            idx_end = i + 1
        elif title_lines:
            break
    return " ".join(title_lines).strip(), idx_end


def split_front_matter(front: list[Line], vocab: set[str], title_end: int) -> tuple[str, int]:
    body = front[title_end:] if title_end <= len(front) else front
    for i, l in enumerate(body):
        if re.match(r"^abstract\b[.:\s]*", l.text, flags=re.I):
            rest = body[i:]
            rest[0] = Line(**{**asdict(rest[0]), "text": re.sub(r"^abstract\b[.:\s]*", "", rest[0].text, flags=re.I)})
            paras = lines_to_paragraphs([x for x in rest if x.text], vocab)
            kept = []
            for p, _ in paras:
                if re.match(r"^(date|homepage|correspondence|keywords?|key words|email|e-mail|index terms)\b", p, flags=re.I):
                    break
                kept.append(p)
            return " ".join(kept), rest[0].page
    paras = lines_to_paragraphs(body, vocab)
    cands = [(p, pg) for p, pg in paras if word_count(p) >= 60]
    if cands:
        p, pg = max(cands, key=lambda x: word_count(x[0]))
        return p, pg
    return "", 1


def build_sections(lines: list[Line], heads: list[Heading], vocab: set[str]):
    #split lines at headings
    if not heads:
        return lines, []
    front = lines[: heads[0].line_index]
    sections: list[Section] = []
    for k, h in enumerate(heads):
        end = heads[k + 1].line_index if k + 1 < len(heads) else len(lines)
        sec = Section(heading=h, lines=lines[h.last_line + 1: end])
        sections.append(sec)
    return front, sections


def apply_exclusions(sections: list[Section]) -> tuple[list[Section], list[str]]:
    #drop references/contents/etc
    kept: list[Section] = []
    removed: list[str] = []
    skipping_toc = False
    skipping_refs = False
    for s in sections:
        h = s.heading
        name = norm_heading_name(h.title)
        if skipping_toc:
            if h.number == "" or re.search(r"\s\d{1,3}$", h.title):
                removed.append(h.title)
                continue
            skipping_toc = False
        if name in ("contents", "table of contents"):
            skipping_toc = True
            removed.append(h.title)
            continue
        if is_excluded_name(h.title):
            removed.append(h.title)
            skipping_refs = name in ("references", "bibliography")
            continue
        skipping_refs = False
        kept.append(s)
    return kept, removed


#transform sections to chunks

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[“\"'])")


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def window_text(paragraphs: list[tuple[str, int]], target=TARGET_WORDS, max_words=MAX_WORDS, overlap=OVERLAP_WORDS) -> list[tuple[str, int, int]]:
    sents: list[tuple[str, int, bool]] = []
    for text, pg in paragraphs:
        ss = split_sentences(text)
        for j, s in enumerate(ss):
            sents.append((s, pg, j == 0))
    windows = []
    cur: list[tuple[str, int, bool]] = []
    cur_w = 0
    for s in sents:
        w = word_count(s[0])
        if w > max_words:
            toks = s[0].split()
            pieces = [" ".join(toks[i:i + target]) for i in range(0, len(toks), target)]
            for pc in pieces:
                if cur and cur_w + word_count(pc) > max_words:
                    windows.append(cur); cur, cur_w = _carry_overlap(cur, overlap)
                cur.append((pc, s[1], False)); cur_w += word_count(pc)
            continue
        if cur and ((cur_w + w > max_words) or (cur_w >= 0.75 * target and s[2]) or cur_w >= target):
            windows.append(cur)
            cur, cur_w = _carry_overlap(cur, overlap)
        cur.append(s); cur_w += w
    if cur:
        new_words = cur_w - _overlap_words(windows[-1], overlap) if windows else cur_w
        if windows and new_words < 0.25 * target and sum(word_count(x[0]) for x in windows[-1]) + new_words <= max_words:
            prev = windows.pop()
            n_carried = len(_carry_overlap(prev, overlap)[0])
            windows.append(prev + cur[n_carried:])
        else:
            windows.append(cur)
    out = []
    for win in windows:
        txt = " ".join(s[0] for s in win)
        out.append((txt, win[0][1], win[-1][1]))
    return out


def _carry_overlap(win, overlap):
    carried, w = [], 0
    for s in reversed(win):
        sw = word_count(s[0])
        if w + sw > overlap * 1.6 and carried:
            break
        carried.insert(0, s); w += sw
        if w >= overlap:
            break
    return carried, w


def _overlap_words(win, overlap):
    return _carry_overlap(win, overlap)[1]


def make_chunks(paper_id: str, title: str, abstract: tuple[str, int],
                sections: list[Section], vocab: set[str], params: dict) -> list[dict]:
    units = []
    if abstract[0]:
        units.append(dict(label="Abstract", number="", path="Abstract",
                          paragraphs=[(abstract[0], abstract[1])], depth=1, top=""))
    stack: list[Heading] = []
    for s in sections:
        h = s.heading
        while stack and stack[-1].depth >= h.depth:
            stack.pop()
        stack.append(h)
        path = " > ".join((f"{x.number} {x.title}".strip()) for x in stack)
        top = stack[0].number.split(".")[0] if stack[0].number else stack[0].title.lower()
        paras = lines_to_paragraphs(s.lines, vocab)
        units.append(dict(label=h.title, number=h.number, path=path, paragraphs=paras,
                          depth=h.depth, top=top))

    #merge tiny units
    def wc(u): return sum(word_count(p) for p, _ in u["paragraphs"])
    def inline_heading(u):
        pg = u["paragraphs"][0][1] if u["paragraphs"] else 0
        return [(f"{u['number']} {u['label']}.".strip(), pg)]

    merged: list[dict] = []
    pending: list[dict] = []
    for idx, u in enumerate(units):
        u = dict(u)
        if pending:
            if pending[0]["top"] == u["top"]:
                pre = []
                covers = []
                for t in pending:
                    if wc(t):
                        pre += inline_heading(t) + t["paragraphs"]
                        covers += t.get("covers", []) + [t["label"]]
                u["paragraphs"] = pre + u["paragraphs"]
                u["covers"] = covers + u.get("covers", [])
            else:
                for t in pending:
                    if wc(t):
                        merged.append(t)
            pending = []
        nxt = units[idx + 1] if idx + 1 < len(units) else None
        is_tiny = wc(u) < params["min_words"] and u["label"] != "Abstract"
        if is_tiny and nxt is not None and nxt["top"] == u["top"]:
            pending.append(u)
            continue
        if is_tiny and merged and merged[-1]["top"] == u["top"] and wc(u):
            prev = merged[-1]
            prev["paragraphs"] = prev["paragraphs"] + inline_heading(u) + u["paragraphs"]
            prev["covers"] = prev.get("covers", []) + [u["label"]]
            continue
        merged.append(u)
    for t in pending:
        if wc(t):
            merged.append(t)

    chunks: list[dict] = []
    for si, u in enumerate(merged):
        total = sum(word_count(p) for p, _ in u["paragraphs"])
        if total == 0:
            continue
        if total <= params["max_words"]:
            wins = [(" ".join(p for p, _ in u["paragraphs"]), u["paragraphs"][0][1], u["paragraphs"][-1][1])]
            method = "section"
        else:
            wins = window_text(u["paragraphs"], params["target_words"], params["max_words"], params["overlap_words"])
            method = "section+window"
        for ci, (txt, p0, p1) in enumerate(wins):
            chunks.append(dict(
                chunk_id=f"{paper_id}_s{si:02d}_c{ci:02d}",
                paper_id=paper_id,
                section_label=u["label"],
                section_number=u["number"],
                section_path=u["path"],
                chunk_index=ci,
                n_chunks_in_section=len(wins),
                chunk_text=txt,
                word_count=word_count(txt),
                page_start=p0, page_end=p1,
                chunking_method=method,
                also_covers="; ".join(u.get("covers", [])),
            ))
    return chunks


def fallback_chunks(paper_id: str, lines: list[Line], vocab: set[str], params: dict) -> list[dict]:
    cut = len(lines)
    for i, l in enumerate(lines):
        if norm_heading_name(l.text) in ("references", "bibliography") and l.bold_frac > 0.5:
            cut = i
            break
    paras = lines_to_paragraphs(lines[:cut], vocab)
    wins = window_text(paras, params["target_words"], params["max_words"], params["overlap_words"])
    return [dict(chunk_id=f"{paper_id}_s00_c{ci:02d}", paper_id=paper_id,
                 section_label="Full text (sections not detected)", section_number="",
                 section_path="Full text", chunk_index=ci, n_chunks_in_section=len(wins),
                 chunk_text=t, word_count=word_count(t), page_start=p0, page_end=p1,
                 chunking_method="fixed_window_fallback", also_covers="")
            for ci, (t, p0, p1) in enumerate(wins)]


#chunking
def chunk_paper(pdf_path: str | Path, title_override: Optional[str] = None,
                min_words=MIN_WORDS, target_words=TARGET_WORDS, max_words=MAX_WORDS,
                overlap_words=OVERLAP_WORDS) -> tuple[dict, list[dict]]:
    #Chunk one paper
    pdf_path = Path(pdf_path)
    params = dict(min_words=min_words, target_words=target_words, max_words=max_words,
                  overlap_words=overlap_words)
    paper_id, version = paper_id_from_name(pdf_path.name)
    lines, diag = extract_lines(pdf_path)
    body_size = diag["body_size"]
    vocab = build_vocab(lines)
    heads = find_headings(lines, body_size)
    title, title_end = _title_from_first_page(lines)

    record = dict(paper_id=paper_id, version=version, title=title_override or title,
                  source_file=pdf_path.name, n_pages=diag["n_pages"],
                  body_font_size=body_size, dropped_lines=diag["dropped"],
                  n_headings_detected=len(heads), warnings=[])
    if not lines:
        record["warnings"].append("no extractable text (scanned PDF?) - needs OCR or abstract-only fallback")
        record["parse_status"] = "failed"
        return record, []

    if len(heads) < FALLBACK_MIN_HEADINGS:
        record["warnings"].append(f"only {len(heads)} headings detected; used fixed-window fallback")
        record["parse_status"] = "fallback"
        return record, fallback_chunks(paper_id, lines, vocab, params)

    front, sections = build_sections(lines, heads, vocab)
    abstract_text, abstract_page = split_front_matter(front, vocab, title_end)
    if any(norm_heading_name(s.heading.title) == "abstract" for s in sections):
        abstract_text = ""
    kept, removed = apply_exclusions(sections)
    record["excluded_sections"] = removed
    record["sections_detected"] = [f"{s.heading.number} {s.heading.title}".strip() for s in kept]
    chunks = make_chunks(paper_id, record["title"], (abstract_text, abstract_page), kept, vocab, params)
    if not abstract_text and not any(norm_heading_name(c["section_label"]) == "abstract" for c in chunks):
        record["warnings"].append("no abstract found in PDF text (use the arXiv API abstract instead)")
    record["n_chunks"] = len(chunks)
    record["parse_status"] = "ok"
    return record, chunks


def chunk_corpus(pdf_dir: str | Path, **kw) -> tuple[list[dict], list[dict]]:
    papers, chunks = [], []
    seen = set()
    for p in sorted(Path(pdf_dir).glob("*.pdf")):
        try:
            rec, ch = chunk_paper(p, **kw)
        except Exception as e:
            pid, ver = paper_id_from_name(p.name)
            papers.append(dict(paper_id=pid, version=ver, source_file=p.name, parse_status="error",
                               warnings=[f"{type(e).__name__}: {e}"], n_chunks=0))
            continue
        if rec["paper_id"] in seen:
            rec["warnings"].append("duplicate paper_id - skipped")
            continue
        seen.add(rec["paper_id"])
        papers.append(rec)
        chunks.extend(ch)
    return papers, chunks



#output
CHUNK_FIELDS = ["chunk_id", "paper_id", "section_label", "chunk_text", "section_number",
                "section_path", "chunk_index", "n_chunks_in_section", "word_count",
                "page_start", "page_end", "chunking_method", "also_covers"]


def write_outputs(papers: list[dict], chunks: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out / "chunks.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHUNK_FIELDS)
        w.writeheader()
        for c in chunks:
            w.writerow({k: c.get(k, "") for k in CHUNK_FIELDS})
    with open(out / "papers.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)


def qa_checks(papers: list[dict], chunks: list[dict], params=None) -> list[str]:
    params = params or dict(max_words=MAX_WORDS, min_words=MIN_WORDS)
    issues = []
    ids = [c["chunk_id"] for c in chunks]
    if len(ids) != len(set(ids)):
        issues.append("duplicate chunk_id values")
    by_paper = collections.defaultdict(list)
    for c in chunks:
        by_paper[c["paper_id"]].append(c)
    for p in papers:
        pid = p["paper_id"]
        if p.get("parse_status") in ("failed", "error"):
            issues.append(f"{pid}: {p.get('parse_status')} - {p.get('warnings')}")
            continue
        cs = by_paper.get(pid, [])
        if not cs:
            issues.append(f"{pid}: produced zero chunks")
        if p.get("parse_status") == "fallback":
            issues.append(f"{pid}: fell back to fixed windows (headings not detected)")
        for w in p.get("warnings", []):
            if "fallback" not in w:
                issues.append(f"{pid}: {w}")
        for c in cs:
            t = c["chunk_text"]
            if c["word_count"] > params["max_words"] * 1.15:
                issues.append(f"{c['chunk_id']}: {c['word_count']} words (> max)")
            if c["word_count"] < 25:
                issues.append(f"{c['chunk_id']}: only {c['word_count']} words")
            if re.search(r"\(cid:\d+\)", t):
                issues.append(f"{c['chunk_id']}: unresolved glyph codes (cid:)")
            if is_excluded_name(c["section_label"]):
                issues.append(f"{c['chunk_id']}: excluded section leaked ({c['section_label']})")
            if re.search(r"arXiv:\d{4}\.\d{4,5}v\d+ \[", t):
                issues.append(f"{c['chunk_id']}: arXiv stamp leaked into text")
        #sanity check
        labels = " ".join(c["section_label"].lower() for c in cs)
        if p.get("parse_status") == "ok" and "introduction" not in labels:
            issues.append(f"{pid}: no 'Introduction' section found")
    return issues


def load_existing(out_dir: Path) -> tuple[list[dict], list[dict]]:
    """Previously written papers/chunks, so --incremental can append instead of rebuilding."""
    try:
        papers = json.loads((out_dir / "papers.json").read_text(encoding="utf-8"))
        with open(out_dir / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return [], []
    return papers, chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk PDFs in data/ into chunks/ for ingestion.")
    parser.add_argument("--pdf-dir", default=str(config.ROOT / "data"))
    parser.add_argument("--out-dir", default=str(config.ROOT / "chunks"))
    parser.add_argument("--incremental", action="store_true",
                        help="Only chunk PDFs whose paper_id isn't already in papers.json; append to existing output")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    old_papers, old_chunks = load_existing(out_dir) if args.incremental else ([], [])
    done = {p["paper_id"] for p in old_papers}

    papers, chunks = [], []
    todo = [p for p in sorted(Path(args.pdf_dir).glob("*.pdf"))
            if paper_id_from_name(p.name)[0] not in done]
    if not todo:
        print("No new PDFs to chunk.")
        return
    # chunk_corpus takes a directory, so run it per-file to honour the incremental filter.
    seen = set()
    for pdf in todo:
        try:
            rec, ch = chunk_paper(pdf)
        except Exception as e:
            pid, ver = paper_id_from_name(pdf.name)
            papers.append(dict(paper_id=pid, version=ver, source_file=pdf.name, parse_status="error",
                               warnings=[f"{type(e).__name__}: {e}"], n_chunks=0))
            continue
        if rec["paper_id"] in seen:
            rec["warnings"].append("duplicate paper_id - skipped")
            continue
        seen.add(rec["paper_id"])
        papers.append(rec)
        chunks.extend(ch)

    issues = qa_checks(papers, chunks)
    write_outputs(old_papers + papers, old_chunks + chunks, out_dir)
    print(f"{len(papers)} papers -> {len(chunks)} chunks ({len(old_papers) + len(papers)} papers total)")
    print("QA:", "all clean" if not issues else f"{len(issues)} issue(s)")
    for issue in issues:
        print("  -", issue)


if __name__ == "__main__":
    main()
