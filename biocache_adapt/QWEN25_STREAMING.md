# Native Qwen2.5-VL streaming adapter

Use `biocache_adapt.QwenStreamAdapter` for the common Qwen2.5-VL backbone. Each fork contains a standalone runtime and a method-specific `qwen_policy.py`. Original upstream and legacy entry points remain available.

## Environment and tests

Python 3.11, PyTorch 2.7.1+cu128, Transformers 4.57.3. GPU inference uses frozen BF16 weights and native SDPA, GQA, vision encoding and MRoPE.

```bash
conda activate biocache
pip install -r biocache_adapt/requirements-qwen.txt
python -m biocache_adapt.test_qwen_stream
```

Tests run a randomly initialized tiny **real Qwen2.5-VL** on CPU. They exercise the native video encoder and decoder, method-specific compression/retrieval, query rollback, causal masks after retrieval, real PTS, odd tails, stream reset, option parsing, dialogue chronology, and storage-budget failures. CPU affinity and frozen weights are checked. Real 7B GPU execution and benchmark quality/latency validation remain pending; these tests establish execution and invariants.

## Usage

```python
from biocache_adapt import QwenStreamAdapter, StreamConfig
adapter = QwenStreamAdapter.load(StreamConfig(
    model_path="/publicdata/model/Qwen2.5-VL-7B-Instruct", device="cuda:0"))
adapter.open_stream("video-id")
adapter.prefill_until(1.0, [(0.0, image0), (1.0, image1)])  # PIL RGB
answer = adapter.answer({"question": "What happened?", "query_time": 1.0})
# Insert only explicitly visible benchmark conversation events:
adapter.inject_dialogue(1.0, "assistant", answer)
report = adapter.close_stream()
adapter.dispose()
```

Prepare ordered `frames.jsonl` (`timestamp`, `frame_path`) and `queries.jsonl` (`query_time`, `question`; optional `query_id`, `options`, `multiple`). Frame paths are relative to the manifest. Timestamps are actual video PTS in seconds; the benchmark driver performs common 1-fps sampling. Adapters consume external frames and never seek source videos.

```bash
python -m biocache_adapt.qwen_stream \
  --model-path /publicdata/model/Qwen2.5-VL-7B-Instruct \
  --device cuda:0 --frames-jsonl frames.jsonl --queries-jsonl queries.jsonl \
  --output outputs/qwen_answers.jsonl --max-new-tokens 256
```

`--config settings.json` overrides `StreamConfig` fields, including method parameters and optional `total_budget_bytes`; its values take precedence over CLI values. Resolved settings appear in every output record.

## Common protocol

- Batch 1, greedy decoding, no repetition penalty, at most 256 new tokens, all native EOS IDs.
- RGB, preserve aspect ratio, maximum 153,664 pixels, dimensions divisible by 28. The minimum representable side is 28 pixels; normal benchmark-sized inputs are downscaled only.
- Consume two visible frames per Qwen temporal patch. At an odd-tail query, privately repeat the last visible frame. Never commit it twice or read the next frame early.
- Spatial positions use native MRoPE. Temporal positions use real first-frame PTS times the model's `tokens_per_second` (2 for the target 7B checkpoint). Retained keys preserve source rotation. Text positions start above the maximum source coordinate.
- Questions are absent from history prefill; query/generated KV is private. Explicit dialogue events are checked against query time and replayed in the query branch. Answers are not automatically added. Judge references and future messages must never be injected.
- LiveVLM/StreamingTOM select history at each native attention layer using its real current Q. A correctly sized causal mask hides later question tokens after selection. Native model forward functions are preserved.
- Restrict the process and existing compute threads to the same maximum two CPU cores; cap PyTorch/BLAS at two threads. External video decoding must apply the same cap.
- Choice parsing requires the exact option set. Ambiguous, invalid and out-of-range outputs remain invalid. Return open responses to the official task scorer unchanged. Benchmark-specific judges and tool environments belong in the benchmark driver.

## Memory and budgets

`report()` separates common model weights, extra static state and persistent history. History includes CPU/GPU KV, archives, representatives, indices, decoded pending frames, metadata and explicit dialogue. Tensor/NumPy backing storage is deduplicated. Python metadata uses a recursive `sys.getsizeof` estimate. Reports also include process RSS/PSS/swap and lifetime peak RSS, disk history (zero here), device CUDA allocated/reserved peaks, phase wall time, observed prepared-vision/query-KV workspace bytes and resolved configuration. `temporary_state()` exposes live query KV and Query observations; operator-internal intermediates are covered by allocator/process peaks. Failed CLI runs write a separate `*.failure.json` record before exiting with an error.

CUDA peaks include model weights and query workspaces. Phase wall time is measured on the host: synchronized CUDA profiling is required for latency claims. Python metadata is an estimate rather than an allocator census.

Defaults track native growth; they are **not equal-total-byte budgets**. Retrieval token limits do not cap CPU/quantized archives. Optional `total_budget_bytes` raises `MemoryError` and records a violation when persistent plus extra static storage exceeds the cap. It does not silently alter method eviction or truncate archives. Calibrate method controls against actual total bytes before comparison and mark over-budget runs as failures. Tiny-model storage numbers are test diagnostics.

## Method integration

Keeps 4 recent temporal pairs as visual features and 16 historical pairs as long-term FIFO KV. Each short-buffer eviction writes one pair to long-term KV. Each query freshly prefills recent features onto a private long-cache copy. The frozen vision encoder runs once per pair; recent features replace the original raw-frame short buffer. Pair counts differ from original single-frame CLI counts. Native source-position MRoPE replaces upstream global forward patches.

## Validation limits

This is a port of upstream mechanisms to a common Qwen backbone. Temporal and positional changes are explicit above; numerical identity and quality parity with original backbones have not been established. Report the backbone and port revision with results. Real-7B GPU validation, official benchmark-driver/scoring execution and equal-byte calibration are subsequent experiment stages.
