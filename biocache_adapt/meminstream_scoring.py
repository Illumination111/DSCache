#!/usr/bin/env python3
"""Offline scoring of DSCache JSONL. Open-answer semantics are not guessed."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import sys

if __package__:
    from .meminstream_official_choices import parse_choice
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from biocache_adapt.meminstream_official_choices import parse_choice


def read_rows(path):
    text = Path(path).read_text(encoding="utf-8")
    return json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()]


def norm(value):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(value)).casefold()))


def choices(value, n):
    # Exactly the released parser, with placeholder option text for labels only.
    return parse_choice(value, [f'__option_{i}__' for i in range(n)])


def binary(value):
    text = norm(value)
    if text in ("yes", "true", "1", "是", "正确"):
        return True
    if text in ("no", "false", "0", "否", "错误"):
        return False
    return None


def objective(row):
    return row.get("question_type") in ("multiple-choice", "multiple-select", "binary") or row.get("original_task") == "Repetition Event Count"


def score(row, response, open_mode, supplied=None):
    kind = row.get("question_type")
    if kind in ("multiple-choice", "multiple-select"):
        opts = row.get("options") or []
        expected = choices(row.get("correct_option"), len(opts))
        if not expected:
            indices = row.get("correct_index")
            indices = indices if isinstance(indices, list) else [indices]
            if indices and all(type(i) is int and 0 <= i < len(opts) for i in indices):
                expected = {chr(65 + i) for i in indices}
        if not expected:
            return dict(correct=None, method="invalid_label")
        option_text = [re.sub(r'^[A-Z][.)]\s*', '', str(x)).strip() for x in opts]
        predicted = parse_choice(response, option_text)
        return dict(correct=predicted == expected, method="option_set",
                    expected=sorted(expected), predicted=sorted(predicted) if predicted else None,
                    format_valid=predicted is not None,
                    dataset_type_label_mismatch=len(expected) > 1 and kind != "multiple-select" and row.get("QA_type") != "mc_multi")
    if kind == "binary":
        expected, predicted = binary(row.get("answer")), binary(response)
        return dict(correct=(predicted == expected) if expected is not None else None,
                    method="binary", format_valid=predicted is not None)
    if row.get("original_task") == "Repetition Event Count":
        expected = re.fullmatch(r"\s*(\d+)\s*", str(row.get("answer", "")))
        predicted = re.fullmatch(r"\s*(\d+)\s*[.。]?\s*", response)
        return dict(correct=(bool(predicted) and int(predicted[1]) == int(expected[1])) if expected else None,
                    method="integer_count", format_valid=predicted is not None)
    exact = norm(response) == norm(row.get("answer", "")) and bool(norm(response))
    if supplied is not None:
        return dict(correct=supplied, method="provided_semantic_score", normalized_exact=exact)
    if open_mode == "exact":
        return dict(correct=exact, method="open_normalized_exact", normalized_exact=exact)
    return dict(correct=None, method="open_needs_semantic_score", normalized_exact=exact)


def stats(items):
    scored = [r for r in items if type(r.get("correct")) is bool]
    completed = sum(r["status"] == "completed" for r in items)
    correct = sum(r["correct"] for r in scored)
    return dict(total=len(items), completed=completed, missing=len(items)-completed,
                coverage=completed/len(items) if items else None, scored=len(scored),
                unscored_completed=completed-len(scored), correct=correct,
                accuracy=correct/len(items) if items else None,
                accuracy_scored=correct/len(scored) if scored else None,
                correct_over_total_lower_bound=correct/len(items) if items else None,
                invalid_format=sum(r.get("format_valid") is False for r in items))


def grouped(items, field):
    groups = defaultdict(list)
    for row in items:
        groups[str(row.get(field, "unknown"))].append(row)
    return {key: stats(value) for key, value in sorted(groups.items())}


def task_macro_report(items, finalized=False):
    """Each level2 task has equal weight within its level1 category.

    All selected samples count in task denominators, including missing/failed
    predictions. Incomplete categories expose a lower bound; their final macro
    accuracy remains null until every sample has a scored prediction or the
    caller explicitly finalizes the run (failures then count as wrong).
    """
    categories = defaultdict(lambda: defaultdict(list))
    for row in items:
        categories[str(row.get('level1') or 'unknown')][str(row.get('level2') or 'unknown')].append(row)
    report = {}
    for category, tasks in sorted(categories.items()):
        task_stats = {}
        for task, rows in sorted(tasks.items()):
            summary = stats(rows)
            summary['accuracy'] = summary['correct_over_total_lower_bound']
            summary['complete'] = all(r['status'] == 'completed' and type(r.get('correct')) is bool for r in rows)
            task_stats[task] = summary
        complete = all(t['complete'] for t in task_stats.values())
        lower = sum(t['accuracy'] for t in task_stats.values()) / len(task_stats)
        report[category] = dict(tasks=task_stats, task_count=len(task_stats),
                                complete=complete, finalized=finalized, macro_accuracy=lower if complete or finalized else None,
                                macro_accuracy_lower_bound=lower,
                                sample_micro=stats([r for rows in tasks.values() for r in rows]))
    return dict(task_field='level2', category_field='level1', finalized=finalized,
                missing_policy='All selected samples in denominator; failed/missing/invalid-format count as wrong. Incomplete running macro is null; --finalized includes failed/missing as wrong.',
                task_count=sum(c['task_count'] for c in report.values()),
                by_level1=report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--finalized", action="store_true", help="Run has ended: missing/failed samples count as wrong in final task macros")
    p.add_argument("--open-ended", choices=("skip", "exact"), default="skip")
    p.add_argument("--open-scores", type=Path, help="Reviewed JSONL: input_index, response, correct (boolean)")
    args = p.parse_args()
    outputs = [args.output_dir / name for name in ("report.json", "details.jsonl", "open_review.jsonl")]
    inputs = [args.input, args.predictions] + ([args.open_scores] if args.open_scores else [])
    if any(a.resolve() == b.resolve() for a in outputs for b in inputs):
        p.error("output paths must not overwrite inputs")
    rows, predictions = read_rows(args.input), {}
    manifest = Path(str(args.predictions) + ".manifest.json")
    if manifest.exists():
        expected_hash = json.loads(manifest.read_text()).get("input_sha256")
        if expected_hash and expected_hash != hashlib.sha256(args.input.read_bytes()).hexdigest():
            p.error("input differs from prediction run manifest")
    for entry in read_rows(args.predictions):
        pred = entry.get("dscache_prediction", {})
        index = pred.get("input_index")
        if type(index) is not int or not 0 <= index < len(rows):
            p.error("prediction has invalid/missing dscache_prediction.input_index")
        if index in predictions:
            p.error(f"duplicate prediction index: {index}; use one merged run")
        for field in ("source_dataset", "source_id", "question", "options", "answer", "query_time", "correct_option", "correct_index"):
            if entry.get(field) != rows[index].get(field):
                p.error(f"prediction/input mismatch at index {index}, field {field}")
        if pred.get("status") != "completed" or not isinstance(pred.get("response"), str):
            p.error(f"index {index}: expected completed prediction with string response")
        predictions[index] = pred
    reviews = {}
    if args.open_scores:
        for entry in read_rows(args.open_scores):
            index = entry.get("input_index")
            if type(index) is not int or index not in predictions or objective(rows[index]):
                p.error("open score must reference a completed non-objective sample")
            if index in reviews or type(entry.get("correct")) is not bool:
                p.error("open scores must be unique with boolean correct")
            if entry.get("response") != predictions[index]["response"]:
                p.error(f"open score response mismatch: {index}")
            reviews[index] = entry["correct"]
    details, pending = [], []
    for index, row in enumerate(rows):
        pred = predictions.get(index)
        detail = {k: row.get(k) for k in ("source_dataset", "source_id", "question_type", "original_task", "level1", "level2")}
        detail.update(input_index=index, objective=objective(row), correct=None,
                      status="completed" if pred else "missing_prediction")
        if pred:
            detail["response"] = pred["response"]
            detail.update(score(row, pred["response"], args.open_ended, reviews.get(index)))
            if not objective(row):
                pending.append(dict(input_index=index, source_dataset=row.get("source_dataset"),
                                    question=row.get("question"), reference=row.get("answer"),
                                    response=pred["response"], correct=reviews.get(index)))
        details.append(detail)
    report = dict(scoring="Unified dataset offline scoring; not the official OVO evaluator",
                  open_ended=args.open_ended, overall=stats(details),
                  objective=stats([r for r in details if r["objective"]]),
                  by_dataset=grouped(details, "source_dataset"),
                  objective_by_dataset=grouped([r for r in details if r["objective"]], "source_dataset"),
                  by_question_type=grouped(details, "question_type"),
                  by_task=grouped(details, "original_task"),
                  task_macro=task_macro_report(details, finalized=args.finalized),
                  by_level1=grouped(details, "level1"), by_level2=grouped(details, "level2"),
                  dataset_type_label_mismatches=sum(r.get("dataset_type_label_mismatch", False) for r in details),
                  notes=["Rates are 0..1. Primary accuracy uses all selected samples; accuracy_scored is diagnostic and excludes missing/unscored rows.",
                         "Missing predictions include failed and not-yet-run samples; these cannot be distinguished from predictions alone.",
                         "correct_over_total_lower_bound counts all missing/unscored as wrong; it is not full semantic accuracy.",
                         "Open exact match is a string metric, not semantic correctness."])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs[0].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    for path, records in zip(outputs[1:], (details, pending)):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(json.dumps({k: report[k] for k in ("overall", "objective", "by_dataset")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
