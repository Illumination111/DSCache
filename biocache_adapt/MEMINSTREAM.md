# DSCache on MemInStream

This entry point ports the decoupled recent-feature / past-KV design to the native
frozen Qwen2.5-VL decoder. It provides a standalone dataset driver, byte budgets,
causal decoding, scoring and resume. No InfiniPot-V checkout is required.

## Environment

Use the dedicated `biocache-baseline-DSCache` conda environment. The local native
port is validated on Python 3.11, Torch 2.9.0+cu128, torchvision 0.24, Transformers
4.57.3 and PyAV 16.0.1. Torch was retained; Transformers was upgraded from 4.49.0
and tokenizers from 0.21.4 to 0.22.2. The original legacy scripts were developed
against the older Transformers stack; keep an older environment when reproducing
those unmodified entry points.

```bash
conda activate biocache-baseline-DSCache
cd /data/user25101366/Baseline-for-BioCache/DSCache
pip install -r biocache_adapt/requirements-qwen.txt
python -m unittest biocache_adapt.test_qwen_stream \
  biocache_adapt.test_meminstream_prompts biocache_adapt.test_meminstream_scoring \
  biocache_adapt.test_meminstream_driver
```

The tiny native-Qwen tests require the target checkpoint's local processor files.
The end-to-end synthetic-video test also requires FFmpeg. It does not load 7B
weights or use a GPU. The 2026-11-17 real-7B acceptance passed on GPU 4; see
`meminstream_validation_20261117.json`. Full benchmark accuracy is still pending.

## Budget definition

Main budget: **1.75 GiB = 1,879,048,192 bytes**. Ablations: 0.875 and 1.3125 GiB.
The budget covers persistent CPU/GPU tensors and arrays, estimated Python metadata
and extra adapter static state. Shared backbone weights are reported separately.

- Four temporal pairs of recent frozen visual features (about 8 sampled seconds).
- Historical BF16 K/V in FIFO order. No unreported CPU or disk archive.
- Both system-prefix copies, three-axis int64 source positions and timestamps.
- One decoded source-size RGB frame awaiting its temporal partner.
- A fixed 64 KiB metadata/static reserve plus 192 bytes per historical pair.

At the first temporal pair of each video, capacity is computed as:

```text
history_pairs = floor((budget - prefix_copies - recent_feature_and_position_capacity
                        - decoded_frame_reserve - fixed_metadata_reserve)
                       / (KV_bytes_per_pair + per_pair_metadata_reserve))
```

The dataset driver sets `decoded_frame_reserve_bytes = source_width*source_height*3`
before opening the stream. Raw-frame bytes are charged as CPU history; source
video files are common inputs, not an offloaded cache. Actual backing storage is
deduplicated, and every stable ingestion/query checks the measured total against
the cap. Insufficient or exceeded budgets raise an error rather than silently
changing the algorithm. Python metadata is an estimate, not an allocator census.

For a representative 192-token temporal unit, 640x480 source video and a 15-token
system prefix, estimated capacities are:

| Total budget | Historical pairs | Recent pairs | Approximate retained time |
| --- | ---: | ---: | ---: |
| 0.875 GiB | 84 | 4 | 176 s |
| 1.3125 GiB | 127 | 4 | 262 s |
| 1.75 GiB | 169 | 4 | 346 s |

Capacity varies with resolution, dtype, source-frame size and prefix length.
`long_pairs` is a legacy limit used only when `total_budget_bytes` is unset.
`cache_tokens` is not a DSCache capacity control. There is no 32K-to-24K TAR step.
Equal maximum bytes do not imply equal average residency; report both.

Queries clone historical KV and freshly prefill recent features. Query KV,
intermediate concatenations and model workspaces are additional temporary memory,
reported separately from the persistent cap. These budgets are not a guarantee
that the entire process fits the same GPU byte count.

## Protocol and commands

Prepare without model loading:

```bash
python -m biocache_adapt.meminstream --prepare-only --device cpu \
  --data-root /data/user25101366/Benchmark-for-BioCache/MemInStream/data/MemInStream \
  --output-dir outputs/meminstream_prepare --budget-gib 1.75
```

This verifies 3,377 videos / 3,795 questions after excluding the two videos longer
than 12 hours. All choice questions now use the released `aligned_prompt_v2_20261020` template:
select every correct option, with unknown answer cardinality. Neither GT values,
label container types nor label lengths influence the request. The released
prompt and choice parser are vendored with source snapshot hashes. Past dialogue
supports both role/content events and question/answer rounds, and excludes events
at or after the query time. Each prediction saves the exact model prompt.

The earlier separated single/multi-choice protocol is superseded; do not resume
its runs with this version. A request or scorer fingerprint mismatch is fatal.

To run later, explicitly select a physical GPU by UUID, after checking resources:

```bash
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_SELECTED_GPU_UUID
export BIOCACHE_PHYSICAL_GPU=REPLACE_WITH_NVIDIA_SMI_INDEX
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m biocache_adapt.meminstream --device cuda:0 \
  --data-root /path/to/MemInStream \
  --output-dir outputs/meminstream_main --budget-gib 1.75 --short-pairs 4
```

For detached execution and 20-minute monitoring, create a fresh run directory and
launch the supervisor from this checkout (replace the environment path if needed):

```bash
mkdir -p outputs/meminstream_main
nohup /data/user25101366/.conda/envs/biocache-baseline-DSCache/bin/python \
  -m biocache_adapt.meminstream_supervise "$PWD/outputs/meminstream_main" \
  /data/user25101366/.conda/envs/biocache-baseline-DSCache/bin/python \
  -m biocache_adapt.meminstream --device cuda:0 --data-root /path/to/MemInStream \
  --output-dir "$PWD/outputs/meminstream_main" --budget-gib 1.75 --short-pairs 4 \
  > outputs/meminstream_main/supervisor.log 2>&1 < /dev/null &
```

The supervisor requires `BIOCACHE_PHYSICAL_GPU`; the worker inherits the UUID.
For resume, wait for the old worker/supervisor to stop and add `--resume` to the
worker command. Do not run two workers on the same output directory.

The shared protocol is 1 Hz latest-visible-frame sampling using actual relative
PTS, maximum 153,664 pixels, greedy generation, native EOS, up to 256 new tokens
and at most two CPU cores. Each question sees only arrived frames/dialogue.
Questions and generated answers are private by default; explicit benchmark
conversation events alone are injected. Independent questions do not share their
reference answers or generated question KV.

After the final question, decoding continues to EOF. An odd last frame is paired
with a private repeat to finalize its visual features. DSCache intentionally keeps
recent features and builds their KV on demand; consuming every sampled frame does
not mean retaining full-video KV or forcing all recent frames into past KV.

## Outputs and recovery

- `manifest.json`: model/settings, source hashes, normalized modes and environment.
- `predictions.jsonl`: original row, `dscache_prediction`, prompt mode,
  `dscache_metrics` and `dscache_state`.
- `video_completion.jsonl`: EOF, decoded and sampled counts, short/long state.
- `memory_after_cleanup.jsonl`: allocated/reserved bytes after GC and empty_cache.
- `progress.json`, `failures.jsonl`, `monitor_20min.jsonl`, `exit_status.json`.
- `execution_revisions.jsonl`: code/settings at resume; old output stays intact.
- `scoring/`: exact option-set accuracy, per-task groups and raw decisions.

Scoring uses the released whole-response choice parser and exact option-set
matching. A prose response is not reduced to its leading letter. Wrong, extra,
missing and unparseable options count as wrong. `report.json` exposes:

- `overall.accuracy`: sample micro accuracy over **all selected rows**.
- `by_level1` / `by_level2`: category/task sample statistics.
- `task_macro.by_level1[category].macro_accuracy`: equal-weight average of the
  category's **level2** task accuracies; the full dataset has 17 such tasks.
  `original_task` is source-dataset taxonomy, not these 17 tasks.

For a running subset, missing predictions count in the displayed lower bounds;
its final macro remains null. The driver passes `--finalized` when execution ends,
so failures/missing rows remain in the final denominator as wrong. Invalid labels
are exposed rather than removed. The result describes the selected dataset only;
a complete 20-question smoke report is not the 3,795-question benchmark score.
Open answers, if added later, require separate semantic scoring.

A nonblocking per-run file lock prevents concurrent workers. Existing outputs
require explicit resume, and fingerprint failures never overwrite saved annotations
or the original manifest. Resume checks annotation/video-metadata/config/protocol fingerprints, skips completed videos, rebuilds
partial-video history while skipping saved answers, prevents duplicate indices,
and preserves an incomplete trailing line separately. GPU OOM is fatal; it is not
silently retried as a different budget. Never resume concurrently.

## Reproduction boundary

This is a **source-position MRoPE Qwen port**, not a claim of exact reproduction of
upstream position-agnostic encoding or `ls_N` historical sampling. Those options
are not silently emulated. The original core paths remain in the repository;
real-7B smoke, memory calibration and paired benchmark comparisons are still
required before publishing conclusions.

## Reproduce the GPU acceptance

The frozen list covers all four categories and 17 level2 tasks, a genuine
multi-answer question, and a query after FIFO eviction. Selection favours short
videos for acceptance speed and is not representative of benchmark accuracy.

```bash
python -m biocache_adapt.meminstream --device cuda:0 \
  --output-dir outputs/meminstream_smoke_FRESH \
  --selection-file biocache_adapt/fixtures/meminstream_smoke20_indices.json \
  --budget-gib 1.75 --short-pairs 4
python -m biocache_adapt.validate_meminstream_gpu --device cuda:0 \
  --output-file outputs/meminstream_smoke_FRESH/gpu_equivalence.json
```

Set `CUDA_VISIBLE_DEVICES` to the intended physical GPU UUID first. The 2026-11-17
run completed 20/20 questions and 20/20 videos, consumed 1,524 sampled frames and
had no failures, invalid outputs or budget violations. Its accuracy was 2/20
(10%); 16 replies selected multiple options. This does not establish an algorithm
ranking or diagnose the cause of low full-benchmark accuracy.

Maximum charged persistent/static state was 1,875,888,374 bytes, below the
1,879,048,192-byte cap across 1,564 stable-state checks. One query followed 17 FIFO
pair evictions. Peak CUDA allocation was 20.69 GiB; peak allocator reservation was
21.13 GiB. Average full-query latency was 0.390 s, maximum 1.271 s. Diagnostic
sampling throughput was about 9.23 frames/s (model load excluded; short-video mix).
CPU affinity was two cores and process swap was zero. GPU work is synchronized
for stage timing; temporary query KV and model workspaces remain outside the
persistent cap and are reported separately.

A separate real-7B no-eviction check compared decoupled prefill with dense
sequential prefill using identical features and source positions. All 28 layers'
K/V and next-token logits had maximum absolute difference 0. Repeated queries
produced identical tokens without changing persistent tensors; a deliberately
insufficient byte cap raised `MemoryError`. This verifies the native port's
internal equivalence, not parity with upstream position-agnostic DSCache.
