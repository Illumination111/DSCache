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
  biocache_adapt.test_meminstream_prompts biocache_adapt.test_meminstream_driver
```

The tiny native-Qwen tests require the target checkpoint's local processor files.
The end-to-end synthetic-video test also requires FFmpeg. It does not load 7B
weights or use a GPU. Actual 7B GPU memory, throughput and accuracy are pending.

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
than 12 hours. The normalized question modes are 3,759 single-choice and 36
multi-select. The dataset labels all rows `multiple-choice`; mode normalization
uses explicit wording/metadata and label *container type*. No label values,
answer content or correct-option count enter the model request. Single-choice
requires one letter; multi-select requests comma-separated letters.

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

Scoring is the same objective option-set rule as the current common driver;
`dataset_type_label_mismatch` flags the dataset's inconsistent type labels, not
an incorrect prompt. It is not presented as the official OVO evaluator. Open
answers, if added later, require separate semantic scoring; they are not guessed.
The scorer is vendored from the local benchmark evaluation utility so a standalone
fork checkout is sufficient. Test scores from random tiny models are not results.

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
