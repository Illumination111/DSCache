"""Native Qwen DSCache: short visual features + bounded long-term KV.

Qwen's temporal patch unit is a pair of frames. Short-term features are freshly
prefilled on each query. Long-term pairs use FIFO eviction as in long_function.
MRoPE positions remain those of the source frames; no global forward patches.
"""
import torch
from .qwen_stream import clone_cache, cache_from, pairs


class Policy:
    name = 'DSCache/Qwen2.5-VL'

    def __init__(self, config):
        self.config = config
        self.reset()

    def reset(self):
        self.cache = None
        self.short = []
        self.tokens_per_pair = None
        self.evictions = 0

    def initialize(self, engine):
        self.cache = clone_cache(engine.system_cache)

    def ingest(self, engine, features, positions, grid):
        if self.tokens_per_pair not in (None, features.shape[1]):
            raise ValueError('DSCache requires constant spatial grid')
        self.tokens_per_pair = features.shape[1]
        self.short.append((features.detach(), positions.detach()))
        while len(self.short) > self.config.short_pairs:
            feats, pos = self.short.pop(0)
            self.cache = engine.forward(feats, self.cache, pos).past_key_values
            prefix = engine.system_cache.get_seq_length()
            limit = self.config.long_pairs * self.tokens_per_pair
            excess = self.cache.get_seq_length() - prefix - limit
            if excess > 0:
                self.cache = cache_from([(torch.cat((k[:, :, :prefix], k[:, :, prefix+excess:]), 2),
                                         torch.cat((v[:, :, :prefix], v[:, :, prefix+excess:]), 2))
                                         for k, v in pairs(self.cache)])
                self.evictions += excess // self.tokens_per_pair

    def query_cache(self, engine, embeddings):
        cache = clone_cache(self.cache)
        for features, positions in self.short:
            cache = engine.forward(features, cache, positions).past_key_values
        return cache

    def state(self):
        return {'long_cache': self.cache, 'short_visual_features': self.short,
                'evicted_pairs': self.evictions, 'tokens_per_pair': self.tokens_per_pair}
