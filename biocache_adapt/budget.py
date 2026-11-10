"""KV 字节预算档配置（协议第 3.2 节）：12.5% / 25% / 50% 三档 + 自定义。

把统一 KV 字节预算映射为 DSCache 的方法超参：
- context_len：累积缓存窗口（单位：帧，官方脚本参数 --context_len）；
- frame_len：短期 buffer 帧数（官方 --frame_len）；
- ltype：长期缓存采样率，'ls_N' 表示每 N 帧取 1（官方 --ltype）；
- stride：最后一帧分辨率倍率（官方 --stride），默认不动。

口径（与官方实现对齐）：
- 原始 KV 字节  = total_frames * tokens_per_frame * bytes_per_token；
- DSCache 持久视觉 KV ≈ (context_len / sample_rate + frame_len) * tokens_per_frame
  * bytes_per_token（长期缓存经 ls_N 采样为 1/N，短期 buffer 为 frame_len 帧；
  system cache 为固定小量，单列不计入档内逼近）。

tokens_per_frame 与模型 KV 结构按 backbone 给预设，也可全部显式覆盖；
预算映射为纯计算，不加载模型、不访问视频。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

BUDGET_TIERS = {"12.5%": 0.125, "25%": 0.25, "50%": 0.5}


@dataclass(frozen=True)
class KVSpec:
    """backbone 的 KV 结构参数。"""

    name: str
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # bf16/fp16
    tokens_per_frame: int = 196  # DSCache llava 路径官方断言每帧 196 token

    @property
    def bytes_per_token(self) -> int:
        # 2 = K + V
        return 2 * self.num_layers * self.num_kv_heads * self.head_dim * self.dtype_bytes


# 官方支持的两种 backbone（数值为公开配置的静态常识，可用参数覆盖）
LLAVA_OV_7B = KVSpec("llava_ov_7b", num_layers=28, num_kv_heads=4, head_dim=128, tokens_per_frame=196)
# qwen 路径每帧 token 数由分辨率决定（默认 448x448 -> 16x16=256），以实际运行校准
QWEN2_5_VL_7B = KVSpec("qwen2_5_vl_7b", num_layers=28, num_kv_heads=4, head_dim=128, tokens_per_frame=256)

PRESETS = {s.name: s for s in (LLAVA_OV_7B, QWEN2_5_VL_7B)}


def parse_ratio(ratio) -> float:
    """接受 0.125 / '0.125' / '12.5%' / '12.5' 等写法。"""
    if isinstance(ratio, (int, float)):
        r = float(ratio)
        return r / 100.0 if r > 1 else r
    s = str(ratio).strip()
    if s in BUDGET_TIERS:
        return BUDGET_TIERS[s]
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    r = float(s)
    return r / 100.0 if r > 1 else r


def raw_kv_bytes(total_frames: int, spec: KVSpec) -> int:
    """原始（未压缩）视频 KV 字节参考值。"""
    if total_frames < 0:
        raise ValueError("total_frames 不能为负")
    return int(total_frames) * spec.tokens_per_frame * spec.bytes_per_token


def dscache_persistent_bytes(context_len: int, frame_len: int, sample_rate: int, spec: KVSpec) -> int:
    """按官方口径估算 DSCache 持久视觉 KV 字节。"""
    kept_frames = context_len // max(1, sample_rate) + frame_len
    return kept_frames * spec.tokens_per_frame * spec.bytes_per_token


def map_budget_to_dscache(
    ratio,
    total_frames: int,
    spec: KVSpec = LLAVA_OV_7B,
    frame_len: int = 4,
    sample_rate: int = 1,
    max_context_len: int = 128,
) -> dict:
    """把 KV 字节预算比例映射为 DSCache 超参。

    由 budget = ratio * total_frames * tpf * bpt 解出
    context_len = clamp((ratio * total_frames - frame_len) * sample_rate)。
    sample_rate（ltype 的 ls_N）作为方法的次压缩旋钮由调用方给定。
    """
    r = parse_ratio(ratio)
    if not 0 < r <= 1:
        raise ValueError(f"预算比例必须在 (0, 1]，收到 {ratio!r}")
    if total_frames <= 0:
        raise ValueError("total_frames 必须为正")
    budget_bytes = int(r * raw_kv_bytes(total_frames, spec))
    context_len = int(round((r * total_frames - frame_len) * sample_rate))
    context_len = max(1, min(context_len, max_context_len, int(total_frames)))
    est = dscache_persistent_bytes(context_len, frame_len, sample_rate, spec)
    return {
        "method": "dscache",
        "budget_ratio": r,
        "total_frames": int(total_frames),
        "spec": spec.name,
        "raw_kv_bytes": raw_kv_bytes(total_frames, spec),
        "budget_bytes": budget_bytes,
        "hyper": {
            "context_len": context_len,
            "frame_len": int(frame_len),
            "ltype": f"ls_{int(sample_rate)}",
            "continuous": True,
        },
        "estimated_persistent_bytes": est,
        "estimated_ratio_achieved": (est / raw_kv_bytes(total_frames, spec)) if total_frames else None,
    }


def standard_budget_mappings(total_frames: int, spec: KVSpec = LLAVA_OV_7B, **kwargs) -> dict:
    """一次返回协议规定的 12.5%/25%/50% 三档。"""
    return {
        tier: map_budget_to_dscache(ratio, total_frames, spec=spec, **kwargs)
        for tier, ratio in BUDGET_TIERS.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="DSCache KV 字节预算档 -> 超参映射（纯计算，不加载模型）")
    parser.add_argument("--ratio", required=True, help="预算档：0.125/0.25/0.5 或 12.5%% 等")
    parser.add_argument("--total-frames", type=int, required=True, help="视频按协议帧率抽帧后的总帧数")
    parser.add_argument("--spec", default="llava_ov_7b", choices=sorted(PRESETS), help="backbone KV 预设")
    parser.add_argument("--frame-len", type=int, default=4)
    parser.add_argument("--sample-rate", type=int, default=1, help="ltype 的 ls_N 采样率 N")
    args = parser.parse_args(argv)
    out = map_budget_to_dscache(args.ratio, args.total_frames, PRESETS[args.spec], args.frame_len, args.sample_rate)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return out


if __name__ == "__main__":
    main()
