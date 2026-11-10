"""统一生成与评分（协议第 3.1/3.3 节）。

- 贪心解码：do_sample=False、num_beams=1、repetition_penalty=1.0，不携带
  temperature/top_p/top_k（对齐协议"生成统一贪心解码"）。
- 选择题首字母匹配：从响应中抽取第一个选项字母与 gt 比较；语义对齐
  DSCache 官方 extract_characters_regex（llava_ov/helper.py）与 OVO-Bench
  官方按字母包含判分，但本地自包含实现，避免给适配层引入官方依赖链。
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

ANSWER_PREFIXES = (
    "the best answer is",
    "the correct answer is",
    "the answer is",
    "answer:",
    "answer",
)


def greedy_gen_kwargs(max_new_tokens: int = 16) -> dict:
    """统一贪心解码参数（协议：repetition_penalty=1.0，沿用百图协议）。"""
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
    }


def extract_first_letter(response: str, valid_letters: Sequence[str] = "ABCDE") -> Optional[str]:
    """选择题首字母匹配抽取。

    优先匹配 "The best answer is X"/"answer is (X)"/"X."/"(X)" 等显式模式；
    否则取响应中第一个独立出现的选项字母；都没有则返回 None（判错）。
    """
    if not response:
        return None
    letters = "".join(valid_letters)
    text = response.strip()
    lower = text.lower()
    # 1. 显式 "answer is X" 类模式
    for prefix in ANSWER_PREFIXES:
        idx = lower.find(prefix)
        if idx != -1:
            tail = text[idx + len(prefix):]
            m = re.search(r"[\(\s\[]*([" + letters + r"])[\)\.\s\]]*", tail)
            if m:
                return m.group(1)
    # 2. (X) / X. / X) 模式
    m = re.search(r"[\(\s\[]?([" + letters + r"])[\)\.][\s]", text + " ")
    if m:
        return m.group(1)
    # 3. 第一个独立字母（前后非字母）
    m = re.search(r"(?<![A-Za-z])([" + letters + r"])[^A-Za-z]", text + " ")
    if m:
        return m.group(1)
    # 4. 单字符响应
    if len(text) == 1 and text in letters:
        return text
    return None


def score_mcq(response: str, gt: str, valid_letters: Sequence[str] = "ABCDE") -> dict:
    """选择题判分：首字母匹配。返回 {pred, correct}。"""
    pred = extract_first_letter(response, valid_letters)
    gt_norm = (gt or "").strip().upper()[:1]
    return {"pred": pred, "gt": gt_norm, "correct": bool(pred) and pred == gt_norm}


def pair_accuracy(records: Sequence[dict]) -> dict:
    """TwinPast PairAcc：同一 pair 的两段历史下都答对方记 1。

    records: [{pair_id, correct}]，按 pair_id 聚合。
    """
    pairs = {}
    for r in records:
        pairs.setdefault(r["pair_id"], []).append(bool(r.get("correct")))
    n = len(pairs)
    n_pass = sum(1 for v in pairs.values() if v and all(v))
    return {"pair_accuracy": (n_pass / n) if n else 0.0, "pairs_pass": n_pass, "pairs_total": n}


def exact_match(response: str, gt: str) -> dict:
    """VNBench 等针捞任务的 EM 判分（归一化后字符串相等）。"""
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().lower())
    return {"pred": norm(response), "gt": norm(gt), "correct": norm(response) == norm(gt)}
