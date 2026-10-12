from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"(?:[^\W_]|[̀-ͯ])+|\S", re.UNICODE)


def dpr_tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFD", str(s)).lower()
    return _TOKEN.findall(s)


def token_key(s: str) -> str:
    return " " + " ".join(dpr_tokens(s)) + " "


def answer_keys(answers: list[str]) -> list[str]:
    keys = []
    for a in answers:
        k = token_key(a)
        if k.strip() and k not in keys:
            keys.append(k)
    return keys


def has_answer_key(text_key: str, ans_keys: list[str]) -> bool:
    return any(k in text_key for k in ans_keys)


def normalize_question(s: str) -> str:
    s = str(s).lower().strip()
    s = s.rstrip("?").strip()
    return " ".join(s.split())


_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september",
           "october", "november", "december"}
_NUMWORDS = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
             "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
             "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
             "thousand", "million", "billion", "first", "second", "third", "fourth", "fifth"}


_CONNECT = {"and", "-", ",", ".", "of", "the", "st", "nd", "rd", "th", "/"}


def answer_type(answers: list[str]) -> str:

    for a in answers:
        toks = dpr_tokens(a)
        if any(any(ch.isdigit() for ch in t) for t in toks):
            return "numdate"
        content = [t for t in toks if t not in _CONNECT]
        if content and all(t in _MONTHS or t in _NUMWORDS for t in content):
            return "numdate"
    return "other"


def is_yesno(answers: list[str]) -> bool:
    return any(a.strip().lower() in {"yes", "no"} for a in answers)
