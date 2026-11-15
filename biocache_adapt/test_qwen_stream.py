"""CPU integration tests using real, randomly initialized tiny Qwen2.5-VL.

Exercises the native vision encoder and decoder, GQA/MRoPE, compression,
retrieval/quantization, repeated query isolation, and streaming tail handling.
No 7B weights or benchmark data are required.
"""
import copy
import json
import unittest
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from .qwen_stream import QwenStreamAdapter, StreamConfig, limit_cpu, pairs, parse_answer, storage_report, retrieval_cache

MODEL = '/publicdata/model/Qwen2.5-VL-7B-Instruct'


def tensors(value):
    from transformers.cache_utils import DynamicCache
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, DynamicCache):
        yield from tensors(pairs(value))
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensors(child)


class NativeQwenTests(unittest.TestCase):
    def setUp(self):
        limit_cpu()
        torch.manual_seed(11)
        cfg = Qwen2_5_VLConfig(
            text_config={'vocab_size': 152064, 'hidden_size': 56, 'intermediate_size': 64,
                         'num_hidden_layers': 2, 'num_attention_heads': 7, 'num_key_value_heads': 1,
                         'max_position_embeddings': 8192, 'use_sliding_window': False,
                         'rope_scaling': {'type': 'mrope', 'mrope_section': [2, 1, 1]}},
            vision_config={'depth': 2, 'hidden_size': 32, 'intermediate_size': 64, 'num_heads': 4,
                           'out_hidden_size': 56, 'tokens_per_second': 2, 'patch_size': 14, 'spatial_merge_size': 2,
                           'temporal_patch_size': 2, 'fullatt_block_indexes': [1], 'window_size': 112},
            image_token_id=151655, video_token_id=151656,
            vision_start_token_id=151652, vision_end_token_id=151653)
        model = Qwen2_5_VLForConditionalGeneration(cfg)
        model.config.text_config._attn_implementation = 'sdpa'
        model.config.vision_config._attn_implementation = 'sdpa'
        model.generation_config.eos_token_id = [151643, 151645]
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, min_pixels=784, max_pixels=12544)
        config = StreamConfig(device='cpu', max_pixels=12544, max_new_tokens=3, recent_tokens=16,
                              retain_tokens=4, group_size=4, retrieval_tokens=8,
                              short_pairs=1, long_pairs=2, granularities=(4, 8, 16),
                              keep_ratios=(.7, .5, .3), granularity_topks=(2, 1, 1))
        self.adapter = QwenStreamAdapter(model, processor, config)
        self.adapter.open_stream('native-integration')
        self.adapter.config.cache_tokens = self.adapter.system_cache.get_seq_length() + 48

    def tearDown(self):
        self.adapter.dispose()

    def test_stream_compress_retrieve_and_query_isolation(self):
        a = self.adapter
        for i in range(16):
            a.push_frames([(float(i), Image.new('RGB', (112, 112), (i*13 % 255, 40, 190)))])
        state = [t.clone() for t in tensors(a.persistent_state())]
        text1 = a.answer('What appears in the video?', query_time=15.)
        tokens1 = list(a.last_answer['token_ids'])
        text2 = a.answer('What appears in the video?', query_time=15.)
        self.assertEqual(tokens1, a.last_answer['token_ids'])
        self.assertEqual(text1, text2)
        after = list(tensors(a.persistent_state()))
        self.assertEqual(len(state), len(after))
        for before, now in zip(state, after):
            self.assertTrue(torch.equal(before, now), 'query mutated persistent tensor')
        self.assertTrue(all(not p.requires_grad for p in a.model.parameters()))
        self.assertLessEqual(len(__import__('os').sched_getaffinity(0)), 2)
        self.assertGreater(a.report()['persistent']['total'], 0)
        self.assertLess(a.report()['static_extra']['total'], 100000)
        if 'InfiniPot' in a.policy.name or 'LiveVLM' in a.policy.name:
            self.assertLessEqual(a.policy.cache.get_seq_length(), a.config.cache_tokens)
        if 'LiveVLM' in a.policy.name:
            self.assertTrue(any(t['available_chunks'] > t['selected_chunks'] > 0
                                for t in a.policy.selection_trace))
        if 'DSCache' in a.policy.name:
            self.assertGreater(a.policy.evictions, 0)
            self.assertEqual(len(a.policy.short), 1)
        if 'MuKV' in a.policy.name:
            self.assertTrue(all(len(x['keys']) > 0 for x in a.policy.store.kv_stores.values()))
        if 'StreamingTOM' in a.policy.name:
            self.assertTrue(a.policy.oqm.quantized_storage['stream'][0])
            self.assertTrue(a.policy.selection_trace)
        with self.assertRaises(ValueError):
            a.answer('Question before known history', query_time=3.)
        with self.assertRaises(ValueError):
            a.push_frames([(15., Image.new('RGB', (112, 112)))])
        a.push_frames([(16., Image.new('RGB', (112, 112)))])
        self.assertEqual(len(a.pending), 1)
        pending = a.pending[0]
        a.answer('Describe the last frame.', query_time=16.)
        self.assertIs(a.pending[0], pending)
        self.assertEqual(a.frames_seen, 17)
        report = a.close_stream()
        self.assertEqual(report['frames_seen'], 17)
        a.open_stream('second-video')
        self.assertEqual(a.frames_seen, 0)
        self.assertEqual(len(a.pending), 0)
        a.close_stream()
        print(json.dumps({'method': a.policy.name, 'validation': 'native tiny Qwen vision + decoder',
                          'persistent_bytes': report['persistent'], 'status': 'passed'}))

    def test_native_retrieval_mask_is_causal(self):
        a = self.adapter
        selected = []
        masks = []
        def select(i, q):
            selected.append((i, q.clone()))
            k, v = pairs(a.system_cache)[i]
            return k[:, :, :2].clone(), v[:, :, :2].clone()
        cache = retrieval_cache(a.system_cache, select)
        def inspect_mask(module, args, kwargs):
            masks.append(kwargs['attention_mask'].clone())
        handle = a.text_model.layers[-1].self_attn.register_forward_pre_hook(inspect_mask, with_kwargs=True)
        try:
            ids = torch.tensor([[50, 51, 52]], device=a.device)
            pos = torch.arange(a.next_position, a.next_position+3).view(1, 1, -1).expand(3, 1, -1)
            output = a.forward(a.embedding(ids), cache, pos)
        finally:
            handle.remove()
        self.assertEqual(len(selected), 2)
        self.assertEqual(output.past_key_values.get_seq_length(), 5)
        self.assertEqual(masks[0].shape, (1, 1, 3, 5))
        self.assertEqual(float(masks[0][0, 0, 0, 2]), 0.)
        self.assertLess(float(masks[0][0, 0, 0, 3]), -1e10)
        self.assertLess(float(masks[0][0, 0, 1, 4]), -1e10)
        self.assertTrue(torch.isfinite(output.last_hidden_state).all())

    def test_dialogue_future_guard_and_total_budget(self):
        a = self.adapter
        a.inject_dialogue(2., 'user', 'Remember the red object.')
        with self.assertRaises(ValueError):
            a.answer('Which object?', query_time=1.)
        a.answer('Which object?', query_time=2.)
        self.assertIsNone(a.temporary_state()['active_query_kv'])
        self.assertEqual(len(a.dialogue_events), 1)
        a.reset()
        self.assertEqual(len(a.dialogue_events), 0)
        self.assertIsNotNone(a.system_cache)
        a.answer('New stream state?', query_time=0.)
        a.config.total_budget_bytes = 1
        with self.assertRaises(MemoryError):
            a.push_frames([(3., Image.new('RGB', (112, 112)))])
        self.assertIsNotNone(a.report()['budget_violation'])

    def test_video_positions_follow_real_pts(self):
        a = self.adapter
        image = Image.new('RGB', (112, 112))
        features, positions, grid = a._prepare_pair([(4., image), (5., image)])
        self.assertEqual(features.shape, (1, 16, 56))
        self.assertTrue((positions[0] == a.video_origin + 8).all())
        self.assertEqual(int(positions[1].min()), a.video_origin)
        self.assertEqual(int(positions[1].max()), a.video_origin + 3)
        self.assertEqual(grid.tolist(), [[1, 8, 8]])

    def test_strict_parser_and_storage_aliases(self):
        self.assertTrue(parse_answer('A,C', ['a', 'b', 'c'], True)['valid'])
        self.assertFalse(parse_answer('A,C', ['a', 'b', 'c'])['valid'])
        self.assertFalse(parse_answer('A or B', ['a', 'b'])['valid'])
        x = torch.zeros(100)
        self.assertEqual(storage_report([x, x[:2]])['cpu'], 400)
        self.assertEqual(storage_report([x.clone(), x.clone()])['cpu'], 800)


if __name__ == '__main__':
    unittest.main()
