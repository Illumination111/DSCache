"""OVO-Bench 评测入口：对齐官方 DSCache/OVOBench_last.py，叠加统一协议。

对齐方式：参数名与官方 argparse 一致（--model_name/--context_len/--frame_len/
--ltype/--continuous/--stride），实际回答调用官方 OVOBench_last.llava_baseline /
qwen_baseline 函数，不复制官方逻辑。

适配层增量：
- --model-path 显式传入模型路径（官方为函数默认值，此处暴露为命令行参数）；
- --budget-ratio 预算档（12.5%/25%/50%），自动映射 context_len/frame_len/ltype
  （映射需要视频帧数，用 --total-frames 代表值；映射结果打印并写入 manifest）；
- --data-root 覆盖官方模块级 TASK_CSV/VIDEO_DIR 常量（monkey-patch 模块属性）；
- 每次运行写 manifest.json 与预算估算 budget_estimate.json；
- --dry-run 不 import 官方模块、不加载模型，只生成 manifest/预算/协议计划。

用法示例：
  python biocache_adapt/run_ovobench.py --model_name llava \
      --model-path /path/to/llava-onevision-qwen2-7b-ov \
      --data-root /path/to/OVO-Bench --budget-ratio 25% --total-frames 600 \
      --run-dir runs/ovobench_llava_25
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biocache_adapt.budget import PRESETS, map_budget_to_dscache
from biocache_adapt.dscache_adapter import DSCacheHyper, setup_official_paths
from biocache_adapt.manifest import collect_manifest, write_manifest
from biocache_adapt.protocol import NullAdapter, SegmentedStreamDriver, StreamQuery, TimestampedFrame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OFFICIAL_ENTRY = os.path.join(REPO_ROOT, "DSCache", "OVOBench_last.py")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OVO-Bench 统一协议入口（对齐官方 OVOBench_last.py）")
    # 与官方一致的参数
    p.add_argument("--model_name", type=str, default="llava", choices=["llava", "qwen"])
    p.add_argument("--context_len", type=int, default=24)
    p.add_argument("--frame_len", type=int, default=4)
    p.add_argument("--ltype", type=str, default="ls_1")
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--stride", type=float, default=1.0)
    # 适配层增量
    p.add_argument("--model-path", type=str, default="", help="模型权重路径；非 dry-run 时必填")
    p.add_argument("--budget-ratio", type=str, default=None, help="KV 预算档，如 12.5%%/25%%/50%%；给定后覆盖 context_len/frame_len/ltype")
    p.add_argument("--total-frames", type=int, default=None, help="预算映射用的代表视频总帧数（1fps）")
    p.add_argument("--sample-rate", type=int, default=1, help="预算映射时 ltype 的 ls_N")
    p.add_argument("--data-root", type=str, default=None, help="OVO-Bench 数据根目录（含 ovo_bench_new.json 与 videos/src_videos）")
    p.add_argument("--segment-seconds", type=float, default=30.0, help="统一协议段长（记录于 manifest）")
    p.add_argument("--run-dir", type=str, default=None, help="运行输出目录（manifest 等）")
    p.add_argument("--dry-run", action="store_true", help="不加载模型、不 import 官方模块")
    return p.parse_args(argv)


def apply_budget(args) -> dict:
    """按预算档映射超参并覆盖 args；返回映射记录。"""
    if not args.budget_ratio:
        return {}
    if not args.total_frames:
        raise SystemExit("--budget-ratio 需要配合 --total-frames（代表视频帧数）")
    spec = PRESETS["llava_ov_7b"] if args.model_name == "llava" else PRESETS["qwen2_5_vl_7b"]
    rec = map_budget_to_dscache(args.budget_ratio, args.total_frames, spec, args.frame_len, args.sample_rate)
    args.context_len = rec["hyper"]["context_len"]
    args.frame_len = rec["hyper"]["frame_len"]
    args.ltype = rec["hyper"]["ltype"]
    args.continuous = True
    return rec


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run:
        raise SystemExit(
            "此适配 CLI 仅提供 --dry-run。真实流式评测必须由外部解码器按时间调用 "
            "DSCacheAdapter.push_frames；禁止回退到官方整视频预读入口。"
        )
    run_dir = args.run_dir or os.path.join(
        "runs", f"ovobench_{args.model_name}_{os.path.basename(args.model_path.rstrip('/'))}"
    )
    os.makedirs(run_dir, exist_ok=True)

    budget_rec = apply_budget(args)
    if budget_rec:
        with open(os.path.join(run_dir, "budget_estimate.json"), "w", encoding="utf-8") as f:
            json.dump(budget_rec, f, ensure_ascii=False, indent=2)

    hyper = DSCacheHyper(
        model_name=args.model_name,
        context_len=args.context_len,
        frame_len=args.frame_len,
        ltype=args.ltype,
        continuous=args.continuous,
        stride=args.stride,
    )
    config = {
        "benchmark": "OVO-Bench",
        "official_entry": os.path.relpath(OFFICIAL_ENTRY, REPO_ROOT),
        "hyper": hyper.to_dict(),
        "model_path": args.model_path,
        "budget": budget_rec or None,
        "segment_seconds": args.segment_seconds,
        "data_root": args.data_root,
        "dry_run": args.dry_run,
    }
    write_manifest(run_dir, collect_manifest(REPO_ROOT, config, official_entry_files=[OFFICIAL_ENTRY]))
    print(f"[biocache_adapt] manifest 已写入 {run_dir}/manifest.json")

    if args.dry_run:
        adapter = NullAdapter()
        query = StreamQuery(1.0, "dry-run question", ["x", "y"], "A", "dry-q")
        events = [
            ("frames", [TimestampedFrame(0.0, None), TimestampedFrame(1.0, None)]),
            ("query", query),
        ]
        result = SegmentedStreamDriver(args.segment_seconds).run_timeline(
            adapter, "dry-stream", events
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "model_loaded": False,
                    "events": adapter.calls,
                    "responses": [response for _, response in result],
                },
                ensure_ascii=False,
            )
        )
        return result

    # ---- 真实运行：import 官方入口并轻封装调用 ----
    setup_official_paths(REPO_ROOT)
    import OVOBench_last  # 官方模块（DSCache/OVOBench_last.py）

    if args.data_root:
        # 官方把数据路径写为模块级常量，运行前替换（不修改官方源码）
        OVOBench_last.TASK_CSV = os.path.join(args.data_root, "ovo_bench_new.json")
        OVOBench_last.VIDEO_DIR = os.path.join(args.data_root, "videos", "src_videos")

    baseline_kwargs = dict(
        ckpt_path=args.model_path,
        context_len=args.context_len,
        frame_len=args.frame_len,
        ltype=args.ltype,
        continuous=args.continuous,
        stride=args.stride,
    )
    if args.model_name == "llava":
        OVOBench_last.llava_baseline(**baseline_kwargs)
    else:
        OVOBench_last.qwen_baseline(**baseline_kwargs)


if __name__ == "__main__":
    main()
