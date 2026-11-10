"""统一流式评测协议：分段流式输入驱动器。

协议来源：《长时流式视频QA对比实验方案》第 3 节：
- 视频切分为固定时长段（默认 30 秒，可配），段内 prefill、段间只允许方法自身
  声明的持久状态传递；
- 问题在其对应时间点到达，此前不得访问未来帧（段内按 query_time 切片推进）；
- 禁止离线全量 prefill 后回答。

本模块只负责"编排"：生成段计划、按 [prefill -> answer -> segment_end] 的事件
序列驱动方法适配器。视频解码、帧编码、缓存操作全部委托给方法适配器（轻封装官
方代码），本模块不复制任何官方逻辑。
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

DEFAULT_SEGMENT_SECONDS = 30.0


@dataclass(frozen=True)
class TimestampedFrame:
    """由评测驱动器提供的单帧；适配器本身不打开或预读视频。"""

    timestamp: float
    frame: Any


@dataclass
class StreamQuery:
    """一个在特定时间点到达的问题。"""

    query_time: float
    question: str
    options: Optional[list] = None
    answer: Optional[str] = None
    query_id: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class SegmentPlan:
    """固定时长段计划。"""

    index: int
    start: float
    end: float
    queries: list = field(default_factory=list)  # list[StreamQuery]


def build_segment_plan(
    duration: float,
    queries: Sequence[StreamQuery],
    segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
) -> list:
    """把视频时长切成固定时长段，并把 query 按 query_time 落入段内。

    返回 list[SegmentPlan]。段边界只决定编排与计量点位，不限制段内按
    query_time 进一步切片 prefill。
    """
    if segment_seconds <= 0:
        raise ValueError("segment_seconds 必须为正")
    n_segments = max(1, int(duration // segment_seconds) + (1 if duration % segment_seconds > 1e-9 else 0))
    plans = [SegmentPlan(i, i * segment_seconds, min((i + 1) * segment_seconds, duration)) for i in range(n_segments)]
    ends = [p.end for p in plans]
    for q in sorted(queries, key=lambda x: x.query_time):
        idx = min(bisect.bisect_left(ends, q.query_time), n_segments - 1)
        plans[idx].queries.append(q)
    return plans


class MethodAdapterProtocol:
    """方法适配器需要实现的接口（鸭子类型，无需显式继承）。

    - open_stream/reset: 建立或重置一个视频流；
    - push_frames(batch): 只接收外部提供的 timestamped frame batch；
    - prefill_until(t, batch): 编码 batch，并断言其中没有 t 之后的帧；
    - answer(query): 在当前持久状态下贪心回答 query，返回 response 文本。
    - close_stream(): 释放流级状态；
    - persistent_state(): 返回 dict[str, 张量/缓存对象]，即"跨段持有的全部
      动态张量"，供双口径计量。只允许这里返回的状态跨段存在。
    """

    def open_stream(self, stream_id: str, **metadata: Any) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def push_frames(self, frames: Iterable[TimestampedFrame], *, visible_until: Optional[float] = None) -> None:
        raise NotImplementedError

    def prefill_until(
        self, t: float, frames: Iterable[TimestampedFrame] = ()
    ) -> None:  # pragma: no cover - 接口声明
        raise NotImplementedError

    def answer(self, query: StreamQuery) -> str:  # pragma: no cover - 接口声明
        raise NotImplementedError

    def persistent_state(self) -> Mapping[str, Any]:  # pragma: no cover - 接口声明
        raise NotImplementedError

    def static_extra_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def temporary_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    def close_stream(self) -> None:
        raise NotImplementedError


class SegmentedStreamDriver:
    """分段流式输入驱动器。

    对每段：先按段内 query 时间交替执行 prefill_until(query_time) / answer，
    段内最后一个 query 之后 prefill_until(segment_end)，然后在段边界调用
    on_segment_end（典型用法：持久张量字节计量）。

    保证：answer(query) 被调用时，适配器只 prefill 过 <= query.query_time 的帧，
    不会访问未来帧。
    """

    def __init__(
        self,
        segment_seconds: float = DEFAULT_SEGMENT_SECONDS,
        on_segment_end: Optional[Callable[[SegmentPlan, MethodAdapterProtocol], None]] = None,
        on_query: Optional[Callable[[StreamQuery, str], None]] = None,
    ):
        if segment_seconds <= 0:
            raise ValueError("segment_seconds 必须为正")
        self.segment_seconds = float(segment_seconds)
        self.on_segment_end = on_segment_end
        self.on_query = on_query

    def run(
        self,
        adapter: MethodAdapterProtocol,
        duration: float,
        queries: Sequence[StreamQuery],
    ) -> list:
        """执行完整流式协议，返回 [(query, response), ...]（按时间序）。"""
        plans = build_segment_plan(duration, queries, self.segment_seconds)
        results = []
        for plan in plans:
            cursor = plan.start
            for q in plan.queries:
                if q.query_time < cursor - 1e-9:
                    raise ValueError(f"query_time {q.query_time} 早于当前流位置 {cursor}，协议违反")
                # 段内 prefill 到 query 时间点（不含未来帧）
                adapter.prefill_until(q.query_time)
                response = adapter.answer(q)
                results.append((q, response))
                if self.on_query is not None:
                    self.on_query(q, response)
                cursor = q.query_time
            # prefill 到段尾（段内无 query 的剩余部分）
            adapter.prefill_until(plan.end)
            if self.on_segment_end is not None:
                self.on_segment_end(plan, adapter)
        return results

    def dry_run_events(self, duration: float, queries: Sequence[StreamQuery]) -> list:
        """不接触模型，仅生成协议事件序列，用于校验编排与数据流。"""
        plans = build_segment_plan(duration, queries, self.segment_seconds)
        events = []
        for plan in plans:
            for q in plan.queries:
                events.append(("prefill_until", round(q.query_time, 6)))
                events.append(("answer", q.query_id or q.question[:32]))
            events.append(("prefill_until", round(plan.end, 6)))
            events.append(("segment_end", plan.index))
        return events

    def run_timeline(
        self,
        adapter: MethodAdapterProtocol,
        stream_id: str,
        events: Iterable[tuple],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> list:
        """执行调用方已按时间排序的 push/query 事件。

        ``("frames", batch)`` 的 batch 必须只含当前已经可见的帧；
        ``("query", StreamQuery)`` 前若曾推入更晚帧，适配器必须拒绝回答。
        本函数不排序、不缓存完整 iterable，因此不会为了调度而预读未来。
        """
        adapter.open_stream(stream_id, **dict(metadata or {}))
        results = []
        try:
            for kind, payload in events:
                if kind == "frames":
                    adapter.push_frames(payload)
                elif kind == "query":
                    response = adapter.answer(payload)
                    results.append((payload, response))
                    if self.on_query is not None:
                        self.on_query(payload, response)
                else:
                    raise ValueError(f"未知 timeline 事件: {kind!r}")
        finally:
            adapter.close_stream()
        return results


class NullAdapter:
    """无模型的参考适配器：只记录协议调用顺序，用于 dry-run 验证驱动器。"""

    def __init__(self):
        self.calls = []
        self.last_timestamp = None

    def open_stream(self, stream_id: str, **metadata: Any) -> None:
        self.calls.append(("open_stream", stream_id))

    def reset(self) -> None:
        self.calls.append(("reset",))
        self.last_timestamp = None

    def push_frames(self, frames, *, visible_until=None) -> None:
        for item in frames:
            timestamp = item.timestamp if hasattr(item, "timestamp") else item[0]
            if visible_until is not None and timestamp > visible_until + 1e-9:
                raise ValueError("frame timestamp 超过 visible_until")
            if self.last_timestamp is not None and timestamp < self.last_timestamp - 1e-9:
                raise ValueError("frame timestamp 必须单调不减")
            self.last_timestamp = float(timestamp)
            self.calls.append(("push_frame", float(timestamp)))

    def prefill_until(self, t: float, frames=()) -> None:
        self.push_frames(frames, visible_until=t)
        self.calls.append(("prefill_until", t))

    def answer(self, query: StreamQuery) -> str:
        if self.last_timestamp is not None and self.last_timestamp > query.query_time + 1e-9:
            raise ValueError("已接收 query_time 之后的未来帧")
        self.calls.append(("answer", query.query_id))
        return "A"  # 占位响应，仅用于协议流验证

    def persistent_state(self) -> Mapping[str, Any]:
        return {}

    def static_extra_state(self) -> Mapping[str, Any]:
        return {}

    def temporary_state(self) -> Mapping[str, Any]:
        return {}

    def close_stream(self) -> None:
        self.calls.append(("close_stream",))
