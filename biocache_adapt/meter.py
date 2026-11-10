"""双口径存储计量工具（协议第 3.2 节）。

- persistent bytes（持久张量）：跨段持有的全部动态张量字节（KV、压缩状态、
  槽位、归档等），历史边界统一为当前问题之前；
- total cost（总成本）：persistent + 方法新增静态参数与缓冲区 + 元数据最小编码。

实现刻意不硬依赖 torch：用鸭子类型识别张量（element_size/nelement）与
DynamicCache（key_cache/value_cache 或 layers[i].keys/values），因此本模块在
无 GPU、无 torch 的环境也可 import 与单元测试。输出为 json。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Optional


def _is_tensor(obj: Any) -> bool:
    return hasattr(obj, "element_size") and hasattr(obj, "nelement") and callable(obj.element_size)


def tensor_nbytes(obj: Any, _seen: Optional[set] = None) -> int:
    """递归统计对象持有的张量字节数。

    支持：torch.Tensor（鸭子类型）、旧版 DynamicCache（.key_cache/.value_cache）、
    新版 DynamicCache（.layers[i].keys/.values）、list/tuple（含 (k, v) 对）、dict。
    用 id 集合防环，同一对象只计一次。
    """
    if obj is None:
        return 0
    if _seen is None:
        _seen = set()
    oid = id(obj)
    if oid in _seen:
        return 0
    if _is_tensor(obj):
        _seen.add(oid)
        return int(obj.element_size()) * int(obj.nelement())
    # 基础类型不计入（避免把 python 容器开销算进张量口径）
    if isinstance(obj, (int, float, str, bool, bytes)):
        return 0
    _seen.add(oid)
    total = 0
    # 旧版 DynamicCache：key_cache / value_cache 列表
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        try:
            for k, v in zip(obj.key_cache, obj.value_cache):
                total += tensor_nbytes(k, _seen) + tensor_nbytes(v, _seen)
            return total
        except TypeError:
            pass
    # 新版 DynamicCache：layers[i].keys / .values
    layers = getattr(obj, "layers", None)
    if layers is not None and not isinstance(obj, (list, tuple, dict)):
        try:
            for layer in layers:
                total += tensor_nbytes(getattr(layer, "keys", None), _seen)
                total += tensor_nbytes(getattr(layer, "values", None), _seen)
            return total
        except TypeError:
            pass
    if isinstance(obj, Mapping):
        for v in obj.values():
            total += tensor_nbytes(v, _seen)
        return total
    if isinstance(obj, (list, tuple)):
        for v in obj:
            total += tensor_nbytes(v, _seen)
        return total
    if is_dataclass(obj) and not isinstance(obj, type):
        for f in fields(obj):
            total += tensor_nbytes(getattr(obj, f.name), _seen)
        return total
    # 仅遍历显式状态对象；调用方不应把整个模型放入计量状态。
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            total += tensor_nbytes(v, _seen)
        return total
    return 0


class StageTimer:
    """可嵌套的分阶段 wall/CUDA 计时器；无 torch 时仍可使用。"""

    def __init__(self):
        self.records = []

    @contextmanager
    def stage(self, name: str, **metadata):
        cuda_start = cuda_end = None
        try:
            import torch

            if torch.cuda.is_available():
                cuda_start = torch.cuda.Event(enable_timing=True)
                cuda_end = torch.cuda.Event(enable_timing=True)
                cuda_start.record()
        except Exception:
            cuda_start = cuda_end = None
        started = time.perf_counter()
        try:
            yield
        finally:
            wall_ms = (time.perf_counter() - started) * 1000.0
            cuda_ms = None
            if cuda_start is not None:
                cuda_end.record()
                cuda_end.synchronize()
                cuda_ms = float(cuda_start.elapsed_time(cuda_end))
            self.records.append(
                {"stage": name, "wall_ms": wall_ms, "cuda_ms": cuda_ms, "metadata": metadata}
            )

    def summary(self) -> dict:
        totals = {}
        for rec in self.records:
            stage = rec["stage"]
            totals.setdefault(stage, {"calls": 0, "wall_ms": 0.0, "cuda_ms": 0.0})
            totals[stage]["calls"] += 1
            totals[stage]["wall_ms"] += rec["wall_ms"]
            if rec["cuda_ms"] is not None:
                totals[stage]["cuda_ms"] += rec["cuda_ms"]
        return {"stages": totals, "records": list(self.records)}


class PersistentBytesMeter:
    """跨段持久状态计量器。

    用法：每段结束时调用 snapshot(adapter.persistent_state())；
    方法新增的静态参数/缓冲区用 register_static 登记一次；
    最后 summary()/dump_json() 输出双口径结果。
    """

    def __init__(self, method_name: str = "", metadata: Optional[dict] = None):
        self.method_name = method_name
        self.metadata = metadata or {}
        self.records = []  # 每次段边界快照
        self.static_tensors = {}  # 方法新增静态参数/缓冲区
        self._peak_gpu_bytes = None

    def snapshot(self, state: Mapping[str, Any], tag: str = "", query_time: Optional[float] = None) -> dict:
        """对当前持久状态做一次字节快照。state: {名称: 张量/缓存对象}。"""
        per_item = {name: tensor_nbytes(obj) for name, obj in (state or {}).items()}
        rec = {
            "tag": tag,
            "query_time": query_time,
            "wall_time": time.time(),
            "per_item_bytes": per_item,
            "persistent_bytes": sum(per_item.values()),
        }
        self.records.append(rec)
        return rec

    def snapshot_adapter(self, adapter: Any, tag: str = "", query_time: Optional[float] = None) -> dict:
        """同时计量 persistent/static/temporary 三类真实张量。"""
        persistent = tensor_nbytes(adapter.persistent_state())
        static = tensor_nbytes(adapter.static_extra_state())
        temporary = tensor_nbytes(adapter.temporary_state())
        rec = {
            "tag": tag,
            "query_time": query_time,
            "wall_time": time.time(),
            "per_item_bytes": {"adapter": persistent},
            "persistent_bytes": persistent,
            "static_extra_bytes": static,
            "temporary_bytes": temporary,
        }
        self.records.append(rec)
        return rec

    def register_static(self, name: str, obj: Any) -> int:
        """登记方法新增的静态参数/缓冲区（非跨段动态状态，但计入总成本）。"""
        n = tensor_nbytes(obj)
        self.static_tensors[name] = n
        return n

    def capture_peak_gpu_memory(self) -> Optional[int]:
        """记录峰值显存（torch.cuda.max_memory_allocated），无 torch 时跳过。"""
        try:
            import torch

            if torch.cuda.is_available():
                self._peak_gpu_bytes = int(torch.cuda.max_memory_allocated())
        except Exception:
            pass
        return self._peak_gpu_bytes

    def _metadata_bytes(self) -> int:
        """元数据最小编码：配置/超参 dict 的紧凑 json utf-8 字节数。"""
        return len(json.dumps(self.metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

    def summary(self) -> dict:
        persistent_series = [r["persistent_bytes"] for r in self.records]
        persistent_last = persistent_series[-1] if persistent_series else 0
        persistent_max = max(persistent_series) if persistent_series else 0
        static_bytes = sum(self.static_tensors.values())
        meta_bytes = self._metadata_bytes()
        return {
            "method": self.method_name,
            "num_snapshots": len(self.records),
            "persistent_bytes_last": persistent_last,
            "persistent_bytes_max": persistent_max,
            "persistent_bytes_series": persistent_series,
            "static_extra_bytes": static_bytes,
            "static_extra_items": dict(self.static_tensors),
            "metadata_min_bytes": meta_bytes,
            # 总成本 = 持久张量 + 方法新增静态参数/缓冲区 + 元数据最小编码
            "total_cost_bytes_last": persistent_last + static_bytes + meta_bytes,
            "total_cost_bytes_max": persistent_max + static_bytes + meta_bytes,
            "peak_gpu_memory_bytes": self._peak_gpu_bytes,
        }

    def dump_json(self, path: str) -> dict:
        summary = self.summary()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "records": self.records}, f, ensure_ascii=False, indent=2)
        return summary
