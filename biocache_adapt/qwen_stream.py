"""Shared native Qwen2.5-VL stream runtime (Transformers 4.57.3).

Distributed in each fork so a standalone checkout is sufficient. Algorithms live
in qwen_policy.py; no global monkey patches or replacement language model.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Set BLAS/tokenizer limits before importing numerical libraries.
for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread_variable] = '2'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb


@dataclass
class StreamConfig:
    model_path: str = '/publicdata/model/Qwen2.5-VL-7B-Instruct'
    device: str = 'cuda:0'
    max_pixels: int = 153664
    max_new_tokens: int = 256
    cpu_threads: int = 2
    total_budget_bytes: int | None = None
    # Algorithm-specific knobs are explicit; these are not equal-byte budgets.
    cache_tokens: int = 6000
    recent_tokens: int = 392
    retain_tokens: int = 50
    retrieval_tokens: int = 12544
    quant_bits: int = 4
    group_size: int = 50
    short_pairs: int = 4
    long_pairs: int = 16
    granularities: tuple = (49, 196, 784)
    keep_ratios: tuple = (.7, .5, .3)
    granularity_topks: tuple = (12, 3, 1)
    scatter_ratio: float = .2
    retrieval_ratio: float = .5

    def __post_init__(self):
        for name in ('max_pixels', 'max_new_tokens', 'cpu_threads', 'cache_tokens',
                     'retain_tokens', 'retrieval_tokens', 'group_size', 'long_pairs'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive')
        if not 1 <= self.cpu_threads <= 2:
            raise ValueError('at most two CPU threads are allowed')
        if self.quant_bits not in (2, 4):
            raise ValueError('quant_bits must be 2 or 4')
        if not 0 < self.retrieval_ratio <= 1 or not 0 <= self.scatter_ratio <= 1:
            raise ValueError('invalid retrieval/scatter ratio')
        if self.total_budget_bytes is not None and self.total_budget_bytes <= 0:
            raise ValueError('total_budget_bytes must be positive')
        if any(g <= 0 for g in self.granularities) or any(not 0 < r <= 1 for r in self.keep_ratios):
            raise ValueError('MuKV granularities and keep ratios must be positive')
        if any(k < 0 for k in self.granularity_topks):
            raise ValueError('MuKV retrieval counts must be nonnegative')
        if self.recent_tokens < 0 or self.short_pairs < 0:
            raise ValueError('recent_tokens and short_pairs must be nonnegative')
        if not (len(self.granularities) == len(self.keep_ratios) == len(self.granularity_topks)):
            raise ValueError('MuKV scale configuration lengths differ')


def limit_cpu(n=2):
    """Pin this process and subsequently spawned workers to at most two CPUs."""
    if not 1 <= n <= 2:
        raise ValueError('CPU limit must be 1 or 2')
    if hasattr(os, 'sched_getaffinity'):
        allowed = sorted(os.sched_getaffinity(0))
        selected = allowed[:n]
        # sched_setaffinity(0) applies only to the calling Linux thread.
        # Restrict pre-existing BLAS/PyTorch threads as well.
        for task in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(task.name), selected)
            except ProcessLookupError:
                pass
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(n)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except ImportError:
        pass


def pairs(cache):
    return [(layer.keys, layer.values) for layer in cache.layers] if cache is not None else []


def cache_from(items):
    cache = DynamicCache()
    for i, (k, v) in enumerate(items):
        cache.update(k, v, i)
    return cache


def clone_cache(cache):
    return cache_from([(k.clone(), v.clone()) for k, v in pairs(cache)])


def retrieval_cache(base, selector):
    """Select history inside each native attention layer, using that layer's Q."""
    cache = clone_cache(base)
    cache.stream_selector = selector
    cache.stream_selected = set()
    cache.stream_extra_tail = None
    return cache


def concat_cache(caches):
    caches = [c for c in caches if c is not None and len(c.layers)]
    if not caches:
        return DynamicCache()
    return cache_from([(torch.cat([pairs(c)[i][0] for c in caches], 2),
                        torch.cat([pairs(c)[i][1] for c in caches], 2))
                       for i in range(len(caches[0].layers))])


def slice_cache(cache, start, stop=None):
    # Clone slices: a view must not retain an unreported full history allocation.
    return cache_from([(k[:, :, start:stop].clone(), v[:, :, start:stop].clone()) for k, v in pairs(cache)])


def storage_report(tree):
    """Deduplicate physical tensor storage, separately by device; count metadata."""
    seen_objects, seen_storage, result = set(), set(), {'cpu': 0, 'cuda': 0, 'metadata': 0}
    def visit(obj):
        if id(obj) in seen_objects:
            return
        seen_objects.add(id(obj))
        if isinstance(obj, torch.Tensor):
            storage = obj.untyped_storage()
            key = (str(obj.device), storage.data_ptr(), storage.nbytes())
            if key not in seen_storage:
                seen_storage.add(key)
                result['cuda' if obj.is_cuda else 'cpu'] += storage.nbytes()
        elif type(obj).__module__ == 'numpy' and hasattr(obj, 'nbytes'):
            base = obj
            while getattr(base, 'base', None) is not None and hasattr(base.base, 'nbytes'):
                base = base.base
            key = ('numpy', base.__array_interface__['data'][0], base.nbytes)
            if key not in seen_storage:
                seen_storage.add(key)
                result['cpu'] += base.nbytes
        elif isinstance(obj, DynamicCache):
            visit(pairs(obj))
        elif isinstance(obj, dict):
            result['metadata'] += __import__('sys').getsizeof(obj)
            for k, v in obj.items():
                visit(k); visit(v)
        elif isinstance(obj, (list, tuple)):
            result['metadata'] += __import__('sys').getsizeof(obj)
            for value in obj:
                visit(value)
        else:
            result['metadata'] += __import__('sys').getsizeof(obj)
    visit(tree)
    result['total'] = sum(result.values())
    return result


def parse_answer(text, options=None, multiple=False):
    """Strict option-set parser; invalid/ambiguous output remains invalid."""
    if not options:
        return {'text': text.strip(), 'valid': bool(text.strip())}
    cleaned = text.strip().upper()
    cleaned = re.sub(r'^(?:ANSWER|答案)\s*[:：]\s*', '', cleaned)
    cleaned = re.sub(r'[().\[\]\s]', '', cleaned)
    if not re.fullmatch(r'[A-Z](?:[,、;/]?[A-Z])*', cleaned):
        return {'text': text, 'valid': False, 'indices': []}
    letters = re.findall('[A-Z]', cleaned)
    indices = sorted(set(ord(x) - 65 for x in letters))
    valid = all(i < len(options) for i in indices) and (multiple or len(indices) == 1)
    return {'text': text, 'valid': valid, 'indices': indices if valid else []}


class QwenStreamAdapter:
    """Question-independent prefill, explicit MRoPE and rollback-only queries."""
    def __init__(self, model, processor, config=None, policy=None):
        self.config = config or StreamConfig()
        limit_cpu(self.config.cpu_threads)
        if model.config.model_type != 'qwen2_5_vl':
            raise ValueError('requires Qwen2.5-VL')
        if model.config.text_config.use_sliding_window:
            raise ValueError('sliding-window backbone is not part of this protocol')
        self.model = model.eval().requires_grad_(False)
        self.processor = processor
        self.text_model = model.model.language_model
        self.embedding = model.get_input_embeddings()
        if policy is None:
            from .qwen_policy import Policy
            policy = Policy(self.config)
        self.policy = policy
        self.stream_id = None
        self._capture = False
        self._queries = {}
        self._hooks = [layer.self_attn.register_forward_pre_hook(self._capture_query, with_kwargs=True)
                       for layer in self.text_model.layers]
        self.reset()

    @classmethod
    def load(cls, config=None):
        config = config or StreamConfig()
        limit_cpu(config.cpu_threads)
        dtype = torch.float32 if config.device == 'cpu' else torch.bfloat16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model_path, dtype=dtype, attn_implementation='sdpa', local_files_only=True)
        model.to(config.device)
        processor = AutoProcessor.from_pretrained(config.model_path, local_files_only=True,
                                                   min_pixels=28 * 28, max_pixels=config.max_pixels)
        return cls(model, processor, config)

    @property
    def device(self):
        return self.embedding.weight.device

    def _capture_query(self, module, args, kwargs):
        cache = kwargs.get('past_key_values')
        selecting = (hasattr(cache, 'stream_selector') and module.layer_idx not in cache.stream_selected)
        if not self._capture and not selecting:
            return
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        query = module.q_proj(hidden).view(hidden.shape[0], hidden.shape[1], -1, module.head_dim).transpose(1, 2)
        cos, sin = kwargs['position_embeddings']
        dummy = query.new_zeros(query.shape[0], module.config.num_key_value_heads, query.shape[2], query.shape[3])
        query, _ = apply_multimodal_rotary_pos_emb(query, dummy, cos, sin, module.rope_scaling['mrope_section'])
        self._queries[module.layer_idx] = query[:, :, -32:].detach()
        if selecting:
            k, v = cache.stream_selector(module.layer_idx, query)
            if cache.stream_extra_tail is not None:
                extra_k, extra_v = pairs(cache.stream_extra_tail)[module.layer_idx]
                k, v = torch.cat((k, extra_k), 2), torch.cat((v, extra_v), 2)
            cache.layers[module.layer_idx].keys = k
            cache.layers[module.layer_idx].values = v
            cache.stream_selected.add(module.layer_idx)
            # The native model prepared its mask before per-layer retrieval.
            # Rebuild it for the actual selected history, preserving causality
            # between question tokens even when the history became shorter.
            length, past = hidden.shape[1], k.shape[2]
            mask = hidden.new_zeros((1, 1, length, past + length))
            tail = torch.full((length, length), torch.finfo(hidden.dtype).min,
                              device=hidden.device, dtype=hidden.dtype).triu(1)
            mask[:, :, :, past:] = tail
            kwargs['attention_mask'] = mask
            return args, kwargs

    def reset(self):
        self._clear_state()
        if self.stream_id is not None:
            self._init_prompt()
            self._check_budget()

    def _clear_state(self):
        if hasattr(self.policy, 'release'):
            self.policy.release()
        self.pending = []
        self.dialogue_events = []
        self.last_timestamp = None
        self.frames_seen = 0
        self.next_position = 0
        self.video_origin = 0
        self.system_cache = None
        self._queries.clear()
        self.last_answer = None
        self.timings = {'vision': 0., 'prefill': 0., 'query': 0.}
        self.policy.reset()
        self.budget_violation = None
        self._active_query_cache = None
        self._workspace_peaks = {'prepared_vision_tensors': 0, 'query_kv': 0}
        self._initialized = False

    def open_stream(self, stream_id='stream', **metadata):
        if self.stream_id is not None:
            raise RuntimeError('close the previous stream first')
        self.reset()
        self.stream_id = str(stream_id)
        self._init_prompt()
        self._check_budget()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        return self.stream_id

    @torch.inference_mode()
    def _init_prompt(self):
        ids = self.processor.tokenizer(
            '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>',
            return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
        pos = torch.arange(ids.shape[-1], device=self.device).view(1, 1, -1).expand(3, 1, -1)
        self.system_cache = self.forward(self.embedding(ids), DynamicCache(), pos).past_key_values
        self.next_position = self.video_origin = ids.shape[-1]
        self.policy.initialize(self)
        self._initialized = True

    @torch.inference_mode()
    def forward(self, embeddings, cache, positions, capture=False, attentions=False):
        self._capture = capture
        self._queries.clear()
        previous = self.model.config.text_config._attn_implementation
        if attentions:
            self.model.config.text_config._attn_implementation = 'eager'
        try:
            return self.text_model(inputs_embeds=embeddings, past_key_values=cache,
                                   position_ids=positions, use_cache=True, output_attentions=attentions,
                                   return_dict=True)
        finally:
            self._capture = False
            self.model.config.text_config._attn_implementation = previous

    @staticmethod
    def _unpack(item):
        if isinstance(item, dict):
            return float(item['timestamp']), item['frame']
        if hasattr(item, 'timestamp'):
            return float(item.timestamp), item.frame
        return float(item[0]), item[1]

    def push_frames(self, frames, visible_until=None):
        if self.stream_id is None:
            raise RuntimeError('open_stream first')
        for item in frames:
            stamp, frame = self._unpack(item)
            if not math.isfinite(stamp) or stamp < 0:
                raise ValueError('timestamps must be finite and nonnegative')
            if visible_until is not None and stamp > visible_until:
                raise ValueError('future frame')
            if self.last_timestamp is not None and stamp <= self.last_timestamp:
                raise ValueError('timestamps must be strictly increasing (no duplicate frames)')
            # Own a compact decoded frame buffer, whose bytes can be counted.
            import numpy as np
            from PIL import Image
            if isinstance(frame, Image.Image):
                frame = np.array(frame.convert('RGB'), copy=True)
            elif isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu()
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = frame.permute(1, 2, 0)
                frame = frame.numpy().copy()
            else:
                frame = np.array(frame, copy=True)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError('frames must be RGB HWC arrays or CHW RGB tensors')
            if frame.dtype != np.uint8:
                if np.issubdtype(frame.dtype, np.floating) and np.isfinite(frame).all() and frame.min() >= 0 and frame.max() <= 1:
                    frame = np.rint(frame * 255).astype(np.uint8)
                else:
                    raise ValueError('frames require uint8 RGB or float RGB in [0, 1]')
            self.pending.append((stamp, frame))
            self.last_timestamp = stamp
            self.frames_seen += 1
            if len(self.pending) == 2:
                features, positions, grid = self._prepare_pair(self.pending)
                start = time.perf_counter()
                self.policy.ingest(self, features, positions, grid)
                self.timings['prefill'] += time.perf_counter() - start
                self.next_position = max(self.next_position, int(positions.max()) + 1)
                self.pending.clear()
            self._check_budget()
        return {'frames': self.frames_seen, 'pending': len(self.pending)}

    def prefill_until(self, timestamp, frames=()):
        self.push_frames(frames, visible_until=timestamp)
        if self.last_timestamp is not None and self.last_timestamp > timestamp:
            raise ValueError('history contains future frames')

    def inject_dialogue(self, timestamp, role, content):
        if self.stream_id is None:
            raise RuntimeError('open_stream first')
        if role not in ('user', 'assistant', 'tool') or not isinstance(content, str):
            raise ValueError('dialogue requires a valid role and text content')
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('invalid dialogue timestamp')
        if self.dialogue_events and timestamp < self.dialogue_events[-1]['timestamp']:
            raise ValueError('dialogue events must be chronological')
        self.dialogue_events.append({'timestamp': float(timestamp), 'role': role, 'content': content})
        self._check_budget()

    @torch.inference_mode()
    def _prepare_pair(self, frames):
        from PIL import Image
        import numpy as np
        converted = []
        for _, frame in frames:
            if isinstance(frame, torch.Tensor):
                frame = frame.detach().cpu().numpy()
                if frame.ndim == 3 and frame.shape[0] == 3:
                    frame = frame.transpose(1, 2, 0)
            frame = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame, dtype=np.uint8))
            frame = frame.convert('RGB')
            w, h = frame.size
            scale = min(1., math.sqrt(self.config.max_pixels / (w * h)))
            target = (max(28, int(w * scale) // 28 * 28), max(28, int(h * scale) // 28 * 28))
            if target[0] * target[1] > self.config.max_pixels:
                raise ValueError('max_pixels too small for image aspect ratio')
            converted.append(frame.resize(target))
        if converted[0].size != converted[1].size:
            raise ValueError('a video must keep a constant decoded frame size')
        start = time.perf_counter()
        batch = self.processor.video_processor(videos=[converted], return_tensors='pt', do_resize=False,
                                                do_sample_frames=False)
        grid = batch['video_grid_thw'].to(self.device)
        pixels = batch['pixel_values_videos'].to(device=self.device, dtype=self.model.dtype)
        features = torch.cat(self.model.model.get_video_features(pixels, grid), 0).unsqueeze(0)
        # Native spatial MRoPE, with the temporal coordinate from real sampled PTS.
        cfg = self.model.config
        ids = torch.tensor([[cfg.vision_start_token_id] + [cfg.video_token_id] * features.shape[1] +
                            [cfg.vision_end_token_id]], device=self.device)
        positions, _ = self.model.model.get_rope_index(ids, video_grid_thw=grid)
        positions = positions[:, :, 1:-1] - 1 + self.video_origin
        tokens_per_second = getattr(self.model.config.vision_config, 'tokens_per_second', 2)
        positions[0] = self.video_origin + round(frames[0][0] * tokens_per_second)
        self.timings['vision'] += time.perf_counter() - start
        prepared = storage_report([pixels, features, positions, grid])
        self._workspace_peaks['prepared_vision_tensors'] = max(
            self._workspace_peaks['prepared_vision_tensors'], prepared['cpu'] + prepared['cuda'])
        return features, positions, grid

    @torch.inference_mode()
    def answer(self, query, query_time=None, options=None, multiple=False):
        if self.stream_id is None:
            raise RuntimeError('open_stream first')
        if not isinstance(query, str):
            getter = query.get if isinstance(query, dict) else lambda name, default=None: getattr(query, name, default)
            query_time = getter('query_time', query_time)
            options = getter('options', options)
            multiple = getter('multiple', multiple)
            query = getter('question')
        if query_time is None:
            query_time = self.last_timestamp if self.last_timestamp is not None else 0.
        if self.last_timestamp is not None and self.last_timestamp > query_time:
            raise ValueError('history contains frames beyond query time')
        if not isinstance(query, str) or not query.strip():
            raise ValueError('question must be nonempty text')
        if self.dialogue_events and self.dialogue_events[-1]['timestamp'] > query_time:
            raise ValueError('dialogue contains future events')
        prompt = query
        if options:
            prompt += '\n' + '\n'.join(f'{chr(65+i)}. {option}' for i, option in enumerate(options))
            prompt += '\nReturn ' + ('all correct option letters, comma-separated.' if multiple else 'only the correct option letter.')
        prefix = '<|vision_end|>\n'
        if self.dialogue_events:
            prefix += '<|im_end|>\n' + ''.join(
                '<|im_start|>' + event['role'] + '\n' + event['content'] + '<|im_end|>\n'
                for event in self.dialogue_events) + '<|im_start|>user\n'
        ids = self.processor.tokenizer(prefix + prompt + '<|im_end|>\n<|im_start|>assistant\n',
                                       return_tensors='pt', add_special_tokens=False)['input_ids'].to(self.device)
        embeddings = self.embedding(ids)
        start = time.perf_counter()
        cache = self.policy.query_cache(self, embeddings)
        self._active_query_cache = cache
        position = self.next_position
        if self.pending:
            # Temporary repeat of the last *visible* frame, never committed to history.
            features, pos, _ = self._prepare_pair([self.pending[0], self.pending[0]])
            if hasattr(cache, 'stream_selector'):
                context = concat_cache([self.policy.tail_context(self), cache.stream_extra_tail])
                tail_output = self.forward(features, context, pos)
                tail = slice_cache(tail_output.past_key_values, -features.shape[1])
                cache.stream_extra_tail = concat_cache([cache.stream_extra_tail, tail])
            else:
                cache = self.forward(features, cache, pos).past_key_values
            position = max(position, int(pos.max()) + 1)
        positions = torch.arange(position, position + ids.shape[-1], device=self.device).view(1, 1, -1).expand(3, 1, -1)
        out = self.forward(embeddings, cache, positions)
        self._active_query_cache = out.past_key_values
        generated = []
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        for i in range(self.config.max_new_tokens):
            token = int(self.model.lm_head(out.last_hidden_state[:, -1]).argmax(-1))
            if token in eos:
                break
            generated.append(token)
            if i + 1 < self.config.max_new_tokens:
                ids = torch.tensor([[token]], device=self.device)
                pos = torch.full((3, 1, 1), position + positions.shape[-1] + i, device=self.device, dtype=torch.long)
                out = self.forward(self.embedding(ids), out.past_key_values, pos)
        self._active_query_cache = out.past_key_values
        query_bytes = storage_report(self._active_query_cache)
        self._workspace_peaks['query_kv'] = max(self._workspace_peaks['query_kv'], query_bytes['cpu'] + query_bytes['cuda'])
        self._active_query_cache = None
        self.timings['query'] += time.perf_counter() - start
        if hasattr(self.policy, 'finish_query'):
            self.policy.finish_query()
        self._queries.clear()
        self._check_budget()
        text = self.processor.tokenizer.decode(generated, skip_special_tokens=True)
        self.last_answer = {'text': text, 'token_ids': generated, 'query_time': query_time,
                            'parsed': parse_answer(text, options, multiple)}
        return text

    query = answer

    def persistent_state(self):
        return {'system_cache': self.system_cache, 'policy': self.policy.state(),
                'pending_frames': self.pending, 'dialogue_events': self.dialogue_events, 'last_timestamp': self.last_timestamp,
                'next_position': self.next_position, 'frames_seen': self.frames_seen}

    def temporary_state(self):
        return {'active_query_kv': self._active_query_cache,
                'observed_workspace_tensor_bytes': dict(self._workspace_peaks),
                'current_layer_query_observations': self._queries}

    def _check_budget(self):
        if self.config.total_budget_bytes is None:
            return
        used = storage_report({'persistent': self.persistent_state(), 'static_extra': self.static_extra_state()})['total']
        if used > self.config.total_budget_bytes:
            self.budget_violation = {'used': used, 'limit': self.config.total_budget_bytes}
            raise MemoryError(f'{self.policy.name}: persistent + extra static bytes {used} exceed total budget '
                              f'{self.config.total_budget_bytes}; do not score this run as budget-compliant')

    def static_extra_state(self):
        return {'adapter_config': asdict(self.config)}

    def report(self):
        process = {}
        try:
            import psutil
            mem = psutil.Process().memory_full_info()
            process = {name: getattr(mem, name, None) for name in ('rss', 'pss', 'swap')}
            import resource
            process['lifetime_peak_rss'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except (ImportError, PermissionError):
            pass
        return {'method': self.policy.name, 'backbone': self.config.model_path,
                'persistent': storage_report(self.persistent_state()),
                'static_extra': storage_report(self.static_extra_state()),
                'temporary': storage_report(self.temporary_state()),
                'observed_workspace_tensor_bytes': dict(self._workspace_peaks),
                'backbone_weight_bytes': storage_report(list(self.model.parameters())),
                'process': process, 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'timing_seconds': dict(self.timings), 'frames_seen': self.frames_seen,
                'cuda_peak_allocated': torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else 0,
                'cuda_peak_reserved': torch.cuda.max_memory_reserved(self.device) if self.device.type == 'cuda' else 0,
                'disk_history_bytes': 0, 'budget_violation': self.budget_violation,
                'budget_mode': 'tracked-growth' if self.config.total_budget_bytes is None else 'strict-total-bytes',
                'metadata_accounting': 'recursive sys.getsizeof estimate; tensor/array backing storage deduplicated',
                'config': asdict(self.config)}

    def close_stream(self):
        report = self.report()
        self.stream_id = None
        self.reset()
        return report

    def dispose(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._queries.clear()
        if hasattr(self.policy, 'release'):
            self.policy.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default=StreamConfig.model_path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', help='JSON object overriding StreamConfig fields')
    parser.add_argument('--frames-jsonl', required=True, help='ordered JSON lines: timestamp, frame_path')
    parser.add_argument('--queries-jsonl', required=True, help='ordered JSON lines: query_time, question, optional options/multiple')
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--cache-tokens', type=int, default=6000)
    args = parser.parse_args()
    config = StreamConfig(model_path=args.model_path, device=args.device,
                          max_new_tokens=args.max_new_tokens, cache_tokens=args.cache_tokens)
    if args.config:
        settings = asdict(config)
        settings.update(json.loads(Path(args.config).read_text()))
        config = StreamConfig(**settings)
    from PIL import Image
    frame_file = Path(args.frames_jsonl)
    with frame_file.open() as source:
        frames = [json.loads(line) for line in source if line.strip()]
    with open(args.queries_jsonl) as source:
        queries = [json.loads(line) for line in source if line.strip()]
    if any(frames[i]['timestamp'] >= frames[i+1]['timestamp'] for i in range(len(frames)-1)):
        raise ValueError('frame manifest must be strictly chronological')
    if any(queries[i]['query_time'] > queries[i+1]['query_time'] for i in range(len(queries)-1)):
        raise ValueError('queries must be chronological')
    adapter = QwenStreamAdapter.load(config)
    index = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    try:
        adapter.open_stream(frame_file.stem)
        with open(args.output, 'w') as output:
            for query in queries:
                while index < len(frames) and frames[index]['timestamp'] <= query['query_time']:
                    frame = frames[index]
                    path = Path(frame['frame_path'])
                    if not path.is_absolute():
                        path = frame_file.parent / path
                    with Image.open(path) as image:
                        adapter.push_frames([(frame['timestamp'], image.convert('RGB').copy())])
                    index += 1
                text = adapter.answer(query)
                output.write(json.dumps({'query_id': query.get('query_id'), 'answer': text,
                                         'parsed': adapter.last_answer['parsed'], 'report': adapter.report()}, ensure_ascii=False) + '\n')
                output.flush()
    except Exception as exc:
        failure = Path(args.output).with_suffix('.failure.json')
        failure.write_text(json.dumps({'status': 'failed', 'error_type': type(exc).__name__,
                                       'error': str(exc), 'report': adapter.report()},
                                      indent=2, ensure_ascii=False) + '\n')
        raise
    finally:
        if adapter.stream_id is not None:
            adapter.close_stream()
        adapter.dispose()


if __name__ == '__main__':
    main()
