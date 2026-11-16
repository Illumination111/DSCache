"""Build model-only requests; normalize the dataset's mislabeled choice types."""
import re


def is_multiselect(row):
    # Dataset question_type is "multiple-choice" even for multi-select samples.
    # Use explicit wording/metadata and label CONTAINER TYPE only, never label
    # values or the number of correct answers in the model prompt.
    return (row.get('question_type') == 'multiple-select'
            or row.get('QA_type') == 'mc_multi'
            or bool(re.search(r'多选|select\s+all|choose\s+all', row['question'], re.I))
            or isinstance(row.get('correct_index'), list)
            or isinstance(row.get('correct_option'), list))


def build_request(row):
    return {'question': row['question'], 'query_time': float(row['query_time']),
            'options': [re.sub(r'^[A-Z][.)]\s*', '', str(x)) for x in (row.get('options') or [])],
            'multiple': is_multiselect(row)}
