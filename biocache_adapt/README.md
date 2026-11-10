# DSCache BioCache streaming adapter

`biocache_adapt` 把 DSCache 接入统一的因果流式视频 QA 协议。视频解码在包外：
调用方按时间顺序传入 `TimestampedFrame`，适配器不会打开视频文件或预读未来帧。

## 接口

```python
adapter.open_stream(video_id)
adapter.push_frames(frame_batch, visible_until=query.query_time)
response = adapter.answer(query)       # `query()` 为别名
adapter.close_stream()
```

同时提供 `reset()`、`prefill_until(t, frame_batch)`、`persistent_state()`、
`static_extra_state()`、`temporary_state()`、真实张量递归计量、分阶段计时、
12.5%/25%/50% 预算映射、manifest 和统一 JSONL schema。通用 loader 接受
`ovobench`、`streamingbench`、`twinpast`、`vnbench`、`egolifeqa`；它只做字段
归一化，不读取视频。

## 支持边界

- LLaVA 路径：帧写入调用官方 `visual_cache_encode_llava_spatial`，回答调用官方
  `OVOBench_last.get_response`。适配层只负责生命周期、因果检查和状态导出。
- Qwen 路径：帧写入连接了官方 `initialize_model_param/visual_cache_encode`；
  官方仓库没有可分离的单 query 回答函数，因此默认 `answer()` 明确报
  `NotImplementedError`。只有调用方显式注入并验证 `answer_func` 后才可使用。
- 未进行模型推理或 benchmark 验证；不要把静态检查视为精度复现。
- `run_ovobench.py` 只允许 `--dry-run`。官方 benchmark 入口会整段解码视频，
  不满足本协议，真实运行必须由外部流式解码器调用上述 API。

## 静态测试

```bash
python -m py_compile biocache_adapt/*.py
python -m biocache_adapt.budget --ratio 25% --total-frames 120
python biocache_adapt/run_ovobench.py --dry-run \
  --budget-ratio 25% --total-frames 120 --run-dir /tmp/dscache-dry
python biocache_adapt/run_ovobench.py --help
```
