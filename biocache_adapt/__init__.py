"""BioCache 对比实验：DSCache baseline 统一流式协议适配层。

与官方代码隔离，只做 import 与轻封装；详见 README.md。
"""

from .budget import BUDGET_TIERS, map_budget_to_dscache, standard_budget_mappings
from .dscache_adapter import DSCacheAdapter, DSCacheHyper
from .manifest import (
    INPUT_SCHEMA_VERSION,
    RESULT_SCHEMA_VERSION,
    append_jsonl,
    load_jsonl,
    normalize_jsonl_record,
    result_record,
)
from .meter import PersistentBytesMeter, StageTimer, tensor_nbytes
from .protocol import (
    NullAdapter,
    SegmentedStreamDriver,
    StreamQuery,
    TimestampedFrame,
)

__all__ = [
    "BUDGET_TIERS",
    "DSCacheAdapter",
    "DSCacheHyper",
    "INPUT_SCHEMA_VERSION",
    "NullAdapter",
    "PersistentBytesMeter",
    "RESULT_SCHEMA_VERSION",
    "SegmentedStreamDriver",
    "StageTimer",
    "StreamQuery",
    "TimestampedFrame",
    "append_jsonl",
    "load_jsonl",
    "map_budget_to_dscache",
    "normalize_jsonl_record",
    "result_record",
    "standard_budget_mappings",
    "tensor_nbytes",
]
