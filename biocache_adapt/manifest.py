"""运行清单（manifest.json）：配置、命令、版本、源码哈希记录（协议第 3.4 节）。

"所有运行保存配置、命令、版本、源码哈希与精简 metrics"。本模块在每次 run
开始时被调用，输出 manifest.json 到 run 目录。所有外部调用（git/包版本）均
容错：缺失时记录为 None 而不是中断运行。
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Optional, Sequence

from .protocol import StreamQuery


INPUT_SCHEMA_VERSION = "biocache.streaming_qa.input.v1"
RESULT_SCHEMA_VERSION = "biocache.streaming_qa.result.v1"
SUPPORTED_DATASETS = ("ovobench", "streamingbench", "twinpast", "vnbench", "egolifeqa")


def sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def hash_sources(paths: Sequence[str]) -> dict:
    """对给定文件/目录计算 sha256；目录递归收集 .py 文件（相对路径为键）。"""
    out = {}
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith(".py"):
                        fp = os.path.join(root, fn)
                        out[os.path.relpath(fp, p)] = sha256_file(fp)
        else:
            out[os.path.basename(p)] = sha256_file(p)
    return out


def git_info(repo_root: str) -> dict:
    def _git(*args):
        try:
            return subprocess.run(
                ["git", "-C", repo_root, *args], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except Exception:
            return None

    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "remote": _git("config", "--get", "remote.origin.url"),
    }


def package_versions(names=("torch", "transformers", "decord", "flash-attn", "qwen-vl-utils")) -> dict:
    from importlib import metadata

    vers = {}
    for n in names:
        try:
            vers[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            vers[n] = None
    return vers


def collect_manifest(
    repo_root: str,
    config: dict,
    official_entry_files: Sequence[str] = (),
    extra: Optional[dict] = None,
) -> dict:
    """汇总一次 run 的完整清单。"""
    adapt_dir = os.path.join(repo_root, "biocache_adapt")
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "python_version": sys.version,
        "config": config,
        "git": git_info(repo_root),
        "package_versions": package_versions(),
        "source_hashes": {
            "biocache_adapt": hash_sources([adapt_dir]),
            "official_entries": hash_sources(list(official_entry_files)),
        },
        "extra": extra or {},
    }


def write_manifest(run_dir: str, manifest: dict) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path


def _first(record: dict, *names, default=None):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return default


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "0").strip()
    parts = text.split(":")
    if len(parts) in (2, 3):
        values = [float(x) for x in parts]
        if len(values) == 2:
            return values[0] * 60 + values[1]
        return values[0] * 3600 + values[1] * 60 + values[2]
    return float(text)


def normalize_jsonl_record(record: dict, benchmark: str) -> dict:
    """把五类 benchmark 的常见字段归一为统一流式 QA schema。

    loader 只解析标注，不打开视频。无法可靠推断的字段保留在 ``meta``，
    不使用 benchmark 私有默认值伪造时间戳。
    """
    benchmark = benchmark.lower()
    if benchmark not in SUPPORTED_DATASETS:
        raise ValueError(f"不支持 benchmark={benchmark!r}; 可选 {SUPPORTED_DATASETS}")
    query_time = _first(record, "query_time", "realtime", "timestamp", "time_stamp", "time")
    if query_time is None:
        raise ValueError("记录缺少 query_time/realtime/timestamp/time_stamp/time")
    options = _first(record, "options", "choices")
    if isinstance(options, dict):
        options = list(options.values())
    answer = _first(record, "answer", "gt", "ground_truth", "label")
    query = StreamQuery(
        query_time=_seconds(query_time),
        question=str(_first(record, "question", "query", "prompt", default="")),
        options=list(options) if options is not None else None,
        answer=None if answer is None else str(answer),
        query_id=str(_first(record, "query_id", "question_id", "id", "qid", default="")),
        meta={"benchmark": benchmark, "source": record},
    )
    return {
        "schema": INPUT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "video_id": str(_first(record, "video_id", "video", "video_name", "clip_id", default="")),
        "video_path": _first(record, "video_path", "path"),
        "duration": _first(record, "duration", "video_duration"),
        "query": query,
    }


def load_jsonl(path: str, benchmark: str) -> list:
    """加载 OVO/StreamingBench/TwinPast/VNBench/EgoLifeQA 通用 JSONL。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(normalize_jsonl_record(json.loads(line), benchmark))
            except Exception as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    return records


def result_record(
    *,
    benchmark: str,
    video_id: str,
    query: StreamQuery,
    response: str,
    metrics: Optional[dict] = None,
) -> dict:
    """构造可直接写入 JSONL 的统一结果记录。"""
    return {
        "schema": RESULT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "video_id": video_id,
        "query": asdict(query),
        "response": response,
        "metrics": metrics or {},
    }


def append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
