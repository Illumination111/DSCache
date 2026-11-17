"""Released MemInStream choice parsing; no leading-letter prose fallback.

Source: HKUST-LongGroup/MemInStream qwen-eval/score_predictions.py
Source snapshot SHA256: 24f9c6bef5c1580c1e5e2d815551ac73d9c3a8fc92e18b901de42fc0c333ff89
"""
import re
import unicodedata

def clean(value):
    if not isinstance(value, (str, bool)):
        return ''
    text = unicodedata.normalize('NFKC', str(value)).strip()
    text = text.replace('**', '').replace('`', '').strip()
    text = re.sub(r'^(?:the\s+answer\s+is|answer\s*[:：]|答案(?:是|为|[:：]))\s*', '', text, flags=re.I)
    return text.strip().strip(' .。!！').strip()

def parse_choice(value, choices):
    """Accept a complete letter/list response or exact option text; never guess from prose."""
    if isinstance(value, list):
        if not value:
            return None
        parsed = [parse_choice(item, choices) for item in value]
        return set().union(*parsed) if all(item is not None for item in parsed) else None
    text = clean(value)
    valid = {chr(65 + i) for i in range(len(choices))}
    letters = text.upper().strip('()[] ')
    if re.fullmatch(r'[A-Z](?:\s*[,、;/]\s*[A-Z])*', letters):
        result = set(re.split(r'\s*[,、;/]\s*', letters))
        return result if result <= valid else None
    # Also accept the exact full option, e.g. "B. Blue", or "Blue".
    matches = set()
    for i, option in enumerate(choices):
        label = chr(65 + i)
        if text.casefold() in {clean(option).casefold(), clean(f'{label}. {option}').casefold(),
                               clean(f'{label}) {option}').casefold()}:
            matches.add(label)
    return matches if len(matches) == 1 else None
