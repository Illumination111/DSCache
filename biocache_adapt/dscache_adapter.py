"""DSCache 方法适配器：对官方代码的 import 与轻封装。

接口点（官方文件/函数）：
- DSCache/llava_ov/llava_onevision.py: Llava_OneVision（llava 路径模型类）
- DSCache/llava_ov/long_function.py: visual_cache_encode_llava_spatial /
  visual_cache_encode / sample_long_cache / concat_dynamic_caches /
  initialize_model_param / TOKEN_IDS（帧编码与缓存操作）
- DSCache/llava_ov/helper.py: prepare_system_cache / clone_dynamic_cache /
  cache_eviction / build_prompt / score / extract_characters_regex
- DSCache/llava_ov/utils.py: StreamingArgs
- DSCache/llava_ov/generate_qwen_op.py: qwen 路径的官方 monkey-patch 函数组
- DSCache/StreamingBench_last.py / OVOBench_last.py: 官方评测入口
  llava_baseline/qwen_baseline 与 llava 路径的 generate_until

官方脚本中 `sys.path.append('ADD your project path to Llava folder here')`
是占位符，本适配层在 import 官方模块前把所需路径注入 sys.path，不修改官方
代码；llava_ov/llava_onevision.py 内部的裸 `import utils` 同理由路径注入解决。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Callable, Iterable, Mapping, Optional

from .meter import StageTimer
from .protocol import StreamQuery, TimestampedFrame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SETUP_DONE = False


def setup_official_paths(repo_root: str = REPO_ROOT) -> None:
    """注入官方代码所需的 import 路径（替代官方脚本里的 sys.path 占位符）。

    - repo_root: llava 包（llava.constants 等）
    - repo_root/DSCache: llava_ov 包与官方评测脚本模块
    - repo_root/DSCache/llava_ov: llava_onevision.py 的裸 `import utils`、
      `from builder import ...`
    """
    global _SETUP_DONE
    if _SETUP_DONE:
        return
    for p in (repo_root, os.path.join(repo_root, "DSCache"), os.path.join(repo_root, "DSCache", "llava_ov")):
        if p not in sys.path:
            sys.path.insert(0, p)
    _SETUP_DONE = True


@dataclass
class DSCacheHyper:
    """DSCache 超参（与官方脚本 argparse 对齐）。"""

    model_name: str = "llava"  # 'llava' | 'qwen'，对应官方 --model_name
    context_len: int = 16  # 累积缓存窗口（帧）
    frame_len: int = 4  # 短期 buffer 帧数
    ltype: str = "ls_1"  # 长期缓存采样：'default' | 'ls_N'
    continuous: bool = True  # 连续位置编码
    stride: float = 1.0  # 最后一帧分辨率倍率

    def to_dict(self) -> dict:
        return asdict(self)


class DSCacheAdapter:
    """DSCache 流式方法适配器。

    llava 路径的回答逻辑完整复用官方 generate_until（见 run_video_llava），
    适配层只负责：模型加载、路径注入、持久状态快照（供双口径计量）、
    按统一协议构造官方 Instance 请求。model_path 通过参数传入，不硬编码。
    """

    def __init__(
        self,
        hyper: Optional[DSCacheHyper] = None,
        repo_root: str = REPO_ROOT,
        answer_func: Optional[Callable[..., str]] = None,
    ):
        self.repo_root = repo_root
        self.hyper = hyper or DSCacheHyper()
        self.model = None
        self.processor = None
        self.stream_args = None
        self._off = None  # 官方模块命名空间缓存
        self._answer_func = answer_func
        self._opened = False
        self._stream_id = None
        self._last_timestamp = None
        self._frames_seen = 0
        self._temporary = {}
        self.timer = StageTimer()

    # ---------------- 模型加载（轻封装官方构造） ----------------
    def load(self, model_path: str, device: str = "cuda") -> None:
        setup_official_paths(self.repo_root)
        if self.hyper.model_name == "llava":
            self._load_llava(model_path, device)
        elif self.hyper.model_name == "qwen":
            self._load_qwen(model_path)
        else:
            raise ValueError(f"未知 model_name: {self.hyper.model_name}")

    def _load_llava(self, model_path: str, device: str) -> None:
        from llava_ov.llava_onevision import Llava_OneVision
        from llava_ov.long_function import visual_cache_encode_llava_spatial

        # 与官方 StreamingBench_last.py/OVOBench_last.py 一致的方法替换
        Llava_OneVision.visual_cache_encode = visual_cache_encode_llava_spatial
        self.model = Llava_OneVision(
            pretrained=model_path,
            conv_template="qwen_1_5",
            model_name="llava_qwen",
            attn_implementation="sdpa",
            device_map=device,
            device=device,
            context_len=self.hyper.context_len,
        )
        self.model.frame_len = self.hyper.frame_len + 1  # 官方脚本同款 +1
        if self._answer_func is None:
            # get_response 是官方可分离的单 query 回答路径；导入发生在 load 后。
            from OVOBench_last import get_response

            self._answer_func = get_response

    def _load_qwen(self, model_path: str) -> None:
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLSdpaAttention,
            Qwen2_5_VLModel,
        )
        from llava_ov.generate_qwen_op import (
            Qwen2_5_VLSdpaAttention_forward,
            Qwen2_5_VLForConditionalGeneration_forward,
            Qwen2_5_VLModel_forward,
            _get_initial_cache_position,
            prepare_inputs_for_generation,
            get_rope_index_multiple,
            _update_model_kwargs_for_generation,
        )
        from llava_ov.utils import StreamingArgs
        from llava_ov.helper import prepare_system_cache

        # 与官方 qwen_baseline 一致的 monkey-patch 组
        Qwen2_5_VLSdpaAttention.forward = Qwen2_5_VLSdpaAttention_forward
        Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
        Qwen2_5_VLForConditionalGeneration.forward = Qwen2_5_VLForConditionalGeneration_forward
        Qwen2_5_VLForConditionalGeneration._get_initial_cache_position = _get_initial_cache_position
        Qwen2_5_VLForConditionalGeneration.prepare_inputs_for_generation = prepare_inputs_for_generation
        Qwen2_5_VLForConditionalGeneration._update_model_kwargs_for_generation = (
            _update_model_kwargs_for_generation
        )
        Qwen2_5_VLForConditionalGeneration.get_rope_index = get_rope_index_multiple

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_path, min_pixels=448 * 448, max_pixels=448 * 448)
        self.stream_args = StreamingArgs(self.model.model.device)
        system_cache, self.stream_args = prepare_system_cache(self.model, self.processor, self.stream_args)
        self.model.system_cache = system_cache
        self.model.context_len = self.hyper.context_len
        self.model.frame_len = self.hyper.frame_len + 1

    # ---------------- 统一流式生命周期 ----------------
    def open_stream(self, stream_id: str, **metadata: Any) -> None:
        if self.model is None:
            raise RuntimeError("请先调用 load()；dry-run 不应调用 open_stream")
        self._stream_id = str(stream_id)
        self.reset()
        self._opened = True

    def reset(self) -> None:
        if self.model is None:
            self._last_timestamp = None
            self._frames_seen = 0
            return
        setup_official_paths(self.repo_root)
        from llava_ov.helper import clone_dynamic_cache

        self.model.visual_cache = clone_dynamic_cache(self.model.system_cache)
        self.model.token_mem = None
        self.model.last_frame = None
        if hasattr(self.model, "wait2cache"):
            self.model.wait2cache = None
        self._last_timestamp = None
        self._frames_seen = 0
        self._temporary = {}

    @staticmethod
    def _unpack_frame(item) -> tuple[float, Any]:
        if isinstance(item, TimestampedFrame):
            return float(item.timestamp), item.frame
        if isinstance(item, Mapping):
            return float(item["timestamp"]), item["frame"]
        timestamp, frame = item
        return float(timestamp), frame

    def push_frames(
        self,
        frames: Iterable[TimestampedFrame],
        *,
        visible_until: Optional[float] = None,
    ) -> None:
        """编码调用方提供的帧批；不打开视频，也不持有未到时间的帧。"""
        if not self._opened:
            raise RuntimeError("流尚未 open_stream")
        setup_official_paths(self.repo_root)
        for item in frames:
            timestamp, frame = self._unpack_frame(item)
            if visible_until is not None and timestamp > visible_until + 1e-9:
                raise ValueError(f"frame {timestamp} 超过 visible_until={visible_until}")
            if self._last_timestamp is not None and timestamp < self._last_timestamp - 1e-9:
                raise ValueError("frame timestamp 必须单调不减")
            with self.timer.stage("video_prefill", timestamp=timestamp):
                if self.hyper.model_name == "llava":
                    self.model.visual_cache_encode(
                        [frame], stride=self.hyper.stride, use_system=False
                    )
                else:
                    from llava_ov.long_function import initialize_model_param, visual_cache_encode

                    # qwen 官方路径以张量帧批为输入；适配器不替调用方解码视频。
                    batch = frame if getattr(frame, "ndim", 0) >= 4 else frame.unsqueeze(0)
                    self.stream_args = initialize_model_param(
                        self.model, self.processor, self.stream_args, batch
                    )
                    self.model.visual_cache, self.stream_args, _ = visual_cache_encode(
                        self.model, self.processor, batch, self.stream_args, use_system=False
                    )
            self._last_timestamp = timestamp
            self._frames_seen += 1

    def prefill_until(self, t: float, frames: Iterable[TimestampedFrame] = ()) -> None:
        """官方编码路径的有界入口；``frames`` 必须全部不晚于 ``t``。"""
        self.push_frames(frames, visible_until=float(t))
        if self._last_timestamp is not None and self._last_timestamp > float(t) + 1e-9:
            raise ValueError("当前缓存已含目标时间之后的未来帧")

    @staticmethod
    def _question_prompt(query: StreamQuery) -> str:
        if not query.options:
            return query.question
        options = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(query.options))
        return (
            f"Question: {query.question}\nOptions:\n{options}\n"
            "Respond only with the letter of the correct option."
        )

    def answer(self, query: StreamQuery) -> str:
        """调用官方单 query 回答函数；查询临时 KV 不写回视频持久缓存。"""
        if not self._opened:
            raise RuntimeError("流尚未 open_stream")
        if self._last_timestamp is not None and self._last_timestamp > query.query_time + 1e-9:
            raise ValueError("拒绝回答：缓存中已含 query_time 之后的未来帧")
        if self._answer_func is None:
            raise NotImplementedError(
                "当前模型路径没有经官方代码分离出的 answer 函数；"
                "请显式传入 answer_func。内置已确认路径仅为 LLaVA get_response。"
            )
        prompt = self._question_prompt(query)
        doc = [
            query.query_id,
            self._stream_id,
            query.meta.get("task", "streaming_qa"),
            prompt,
            query.query_time,
            query.answer,
        ]
        with self.timer.stage("query_answer", query_id=query.query_id):
            response = self._answer_func(
                model=self.model,
                doc=doc,
                frames=self._frames_seen > 0,
                ltype=self.hyper.ltype,
                continuous=self.hyper.continuous,
            )
        return str(response)

    query = answer

    def close_stream(self) -> None:
        if self.model is not None:
            for name in ("visual_cache", "token_mem", "last_frame", "wait2cache"):
                if hasattr(self.model, name):
                    setattr(self.model, name, None)
        self._opened = False
        self._stream_id = None
        self._last_timestamp = None
        self._frames_seen = 0
        self._temporary = {}

    # ---------------- 持久状态（双口径计量的"持久张量"来源） ----------------
    def persistent_state(self) -> Mapping[str, Any]:
        """DSCache 声明的跨段持久状态：system_cache + visual_cache + 帧缓冲。

        对应官方脚本里每个视频/每次查询间被保留的模型属性
        （StreamingBench_last.py 中 self.visual_cache/self.token_mem/self.last_frame，
        qwen 路径另有 wait2cache）。
        """
        state = {}
        for name in ("system_cache", "visual_cache", "token_mem", "last_frame", "wait2cache"):
            if hasattr(self.model, name):
                state[name] = getattr(self.model, name)
        return state

    @staticmethod
    def static_extra_state() -> Mapping[str, Any]:
        """DSCache 为免训练方法，无新增静态参数/缓冲区。"""
        return {}

    def temporary_state(self) -> Mapping[str, Any]:
        """查询临时态；官方 get_response 内的局部 clone 不跨调用保留。"""
        return dict(self._temporary)

    # ---------------- llava 路径：复用官方 generate_until 的整视频驱动 ----------------
    def run_video_llava(self, video_path: str, queries, output_jsonl: str,
                        generate_until_func=None, task_name: str = "twinpast") -> list:
        """用官方 StreamingBench_last.generate_until 回答一个视频的全部 query。

        queries: list[(query_time_sec, ctx_text, doc)]，doc 为官方输出记录格式
        [question_id, task_type, question, time_stamp, answer, options, ...]。
        generate_until_func 默认取官方 StreamingBench_last.generate_until，
        也可传 OVOBench 变体；回答逻辑 100% 为官方代码。
        """
        setup_official_paths(self.repo_root)
        from llava_ov.builder import Instance
        from llava_ov.llava_onevision import Llava_OneVision

        if generate_until_func is None:
            import StreamingBench_last

            generate_until_func = StreamingBench_last.generate_until
        Llava_OneVision.generate_until = generate_until_func

        from .scoring import greedy_gen_kwargs

        gen_kwargs = greedy_gen_kwargs(max_new_tokens=16)
        gen_kwargs["until"] = ["\n\n"]  # 官方脚本惯例
        requests = {video_path: []}
        for query_time, ctx, doc in queries:
            doc_to_visual = ([video_path], float(query_time), 1)  # ([path], t_sec, fps)
            arguments = (ctx, dict(gen_kwargs), doc_to_visual, doc[0], task_name, "test")
            inst = Instance(
                request_type="generate_until",
                arguments=arguments,
                idx=0,
                doc=doc,
                metadata={"task": task_name, "doc_id": doc[0], "repeats": 1, "split": "test"},
            )
            requests[video_path].append(inst)
        requests[video_path].sort(key=lambda i: i.arguments[2][1])
        return self.model.generate_until(
            requests, output_jsonl, self.hyper.ltype, self.hyper.continuous, self.hyper.stride
        )
