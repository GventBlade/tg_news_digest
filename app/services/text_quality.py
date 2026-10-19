"""Dependency-free checks for editorial quality of automatically generated news.

These checks are intentionally narrow: they catch obvious unusable headings and
repeated sentences. Semantic accuracy is handled separately by Final Fact-Check.
Manual administrator news is excluded by callers.
"""

import html
import re
from difflib import SequenceMatcher
from typing import List


def _plain(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", str(value or ""))).strip()


def _norm(value: str) -> str:
    return re.sub(r"[^\w]+", " ", _plain(value).casefold(), flags=re.UNICODE).strip()


def _sentence_parts(text: str) -> List[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+", _plain(text))
        if part.strip()
    ]


def weak_auto_headline(headline: str) -> bool:
    clean = _norm(headline)
    clean = re.sub(r"^(?:оновлено|update)\s+", "", clean)
    if not clean:
        return True
    generic = {
        "жах", "шок", "жесть", "терміново", "увага", "дивіться",
        "ось це так", "страшні кадри", "ексклюзив", "сенсація",
        "кадри з місця події", "важлива новина", "неймовірно",
    }
    if clean in generic:
        return True
    words = clean.split()
    # Clickbait beginning with a vague interjection is not informative.
    if words and words[0] in {"жах", "шок", "жесть", "сенсація"} and len(words) <= 5:
        return True
    # A single content-free word should never be accepted as a news headline.
    # Keep two-word titles like 'Помер актор' or 'Ухвалено закон' valid.
    return len(words) <= 1 and len(clean) < 18


def _near_duplicate(left: str, right: str) -> bool:
    a, b = _norm(left), _norm(right)
    if not a or not b:
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 26:
        return False
    # Catch light paraphrasing within one post without removing distinct facts.
    return SequenceMatcher(None, a, b).ratio() >= 0.94


def deduplicate_fallback_sentences(sentences: List[str]) -> List[str]:
    """Remove literal/near-literal repeats and empty video-caption boilerplate."""
    clean = []
    for sentence in sentences:
        candidate = re.sub(r"\s+", " ", str(sentence or "")).strip()
        plain = _norm(candidate)
        if not plain:
            continue
        if plain in {
            "деталі та наслідки", "подробиці згодом", "подробиці уточнюються",
            "кадри публікують місцеві змі", "дивіться відео", "кадри публікують змі",
            "прямі кадри конкретного удару або його безпосередніх наслідків",
        }:
            continue
        if any(_near_duplicate(candidate, old) for old in clean):
            continue
        clean.append(candidate)
    return clean


def auto_news_quality_issues(text: str) -> List[str]:
    """Report only high-confidence issues; does not silently rewrite facts."""
    raw = str(text or "")
    title_match = re.search(r"<b>(.*?)</b>", raw, re.I | re.S)
    if not title_match:
        return ["missing_bold_headline"]
    headline = title_match.group(1)
    body = raw[title_match.end():]
    issues = []
    if weak_auto_headline(headline):
        issues.append("vague_or_clickbait_headline")
    sentences = _sentence_parts(body)
    if not sentences or not _norm(body):
        issues.append("empty_body")
    else:
        seen = []
        for sentence in sentences:
            if len(_norm(sentence)) < 24:
                continue
            if any(_near_duplicate(sentence, old) for old in seen):
                issues.append("repeated_sentences")
                break
            seen.append(sentence)
    return issues
