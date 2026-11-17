"""Official unknown-cardinality choice prompt, independent of answer labels."""
from .meminstream_official_prompt import make_prompt, options, seconds, PROMPT_VERSION


def build_request(row, start=0., end=None):
    # Only model-visible fields enter the released prompt. GT, evidence and task
    # metadata are never used, including their types or lengths.
    visible = {key: row[key] for key in
               ('question', 'query_time', 'options', 'dialogue_events', 'dialogue_history', 'question_type')
               if key in row}
    query_time = seconds(visible['query_time'])
    end = query_time if end is None else float(end)
    if not 0 <= start <= end <= query_time:
        raise ValueError('invalid causal prompt interval')
    return {'question': visible['question'], 'query_time': query_time,
            'options': options(visible), 'multiple': True,
            'prompt': make_prompt(visible, float(start), end),
            'prompt_version': PROMPT_VERSION}
