# Vendored from HKUST-LongGroup/MemInStream qwen-eval/prompt_templates.py.
# Source snapshot SHA256: d2366ce4cd9adef402f4e4b2a5f575a40ace3ac24035fa1644f3fac4b38a8052
"""Final evaluation prompt shared across methods."""
import math
import re


def seconds(value):
    if value is None or isinstance(value, bool):
        raise ValueError(f"invalid time: {value!r}")
    parts = str(value).strip().replace("：", ":").split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"invalid time: {value!r}")
    number = sum(float(x) * (60 ** i) for i, x in enumerate(reversed(parts)))
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"invalid time: {value!r}")
    return float(number)

def options(row):
    return [re.sub(r"^[A-Z][.)]\s*", "", str(x)).strip() for x in row.get("options") or []]

def dialogue_prompt(row, query_time):
    events = row.get('dialogue_events', row.get('dialogue_history'))
    if not events:
        return ''
    if not isinstance(events, list):
        raise ValueError('dialogue history must be a list')
    history = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('invalid dialogue event')
        timestamp = seconds(event.get('time', event.get('timestamp', event.get('query_time'))))
        # Only prior dialogue is available, regardless of the video clip window.
        if timestamp >= query_time:
            continue
        label = f'{event["event_id"]} ' if event.get('event_id') else ''
        if event.get('role'):
            role, content = event['role'], event.get('content')
            if role not in {'user', 'assistant'} or not isinstance(content, str) or not content.strip():
                raise ValueError('history requires user/assistant roles and nonempty content')
            text = f'{label}[{timestamp:.3f}s] {role}: {content}'
        else:
            question, answer = event.get('question'), event.get('answer')
            if not isinstance(question, str) or not question.strip() or answer is None:
                raise ValueError('history round requires a question and its correct answer')
            text = f'{label}[{timestamp:.3f}s] user: {question}'
            choices = options(event)
            if choices:
                text += '\n' + '\n'.join(f'{chr(65+i)}. {value}' for i, value in enumerate(choices))
            text += f'\n{label}[{timestamp:.3f}s] assistant: {answer}'
        history.append((timestamp, text))
    if not history:
        return ''
    return ('Previous dialogue (assistant replies are correct answers; times are original-video times):\n'
            + '\n'.join(text for _, text in sorted(history, key=lambda item: item[0])) + '\n\n')

# The released prompt is the final template (formerly v2).
PROMPT_VERSION = 'aligned_prompt_v2_20261020'


def prompt_version():
    """Identifier recorded in prediction metadata for reproducibility."""
    return PROMPT_VERSION


def choice_instruction():
    return ('Select all correct options. Select multiple options only when each is independently '
            'supported by the video and any provided dialogue history. '
            'Do not select additional options merely because you are uncertain. '
            'Your answer is correct only if the selected options exactly match all correct options; '
            'selecting any incorrect option or missing any correct option makes the answer incorrect. '
            'Return only the selected option letter(s), separated by commas, without explanation.')


def make_prompt(row, start, end):
    question = row.get('question', row.get('Question'))
    if not isinstance(question, str) or not question.strip():
        raise ValueError('missing nonempty question')
    query_time = seconds(row.get('query_time', row.get('Question time')))
    prompt = (f'The supplied video shows original-video time {start:.3f}s to {end:.3f}s. '
              f'The question is asked at {query_time:.3f}s. '
              f'Video timestamps are clip-local; add {start:.3f}s for original-video time.\n')
    prompt += dialogue_prompt(row, query_time)
    prompt += f'Question: {question}\n'
    choices = options({'options': row.get('options', row.get('Options'))})
    if choices:
        if len(choices) > 26:
            raise ValueError('at most 26 answer options are supported')
        prompt += '\n'.join(f'{chr(65+i)}. {value}' for i, value in enumerate(choices))
        prompt += '\n' + choice_instruction()
    elif row.get('question_type') == 'binary':
        prompt += 'Return only True or False.'
    else:
        prompt += 'Give a concise, complete answer in the same language as the question. Return only the answer.'
    return prompt
