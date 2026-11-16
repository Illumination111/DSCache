"""DSCache Qwen port: on-demand recent prefill + byte-bounded FIFO past KV.

Source-position native MRoPE is an explicit porting choice. This does not claim
parity with the upstream position-agnostic encoding and optional ls_N sampling.
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
        self.long_times = []
        self.tokens_per_pair = None
        self.evictions = 0
        self.capacity = None
        self.kv_bytes_per_token = None
        self.budget_details = {}

    def initialize(self, engine):
        self.cache = clone_cache(engine.system_cache)
        length = engine.system_cache.get_seq_length()
        self.kv_bytes_per_token = sum((k.numel()*k.element_size()+v.numel()*v.element_size())//length
                                     for k,v in pairs(engine.system_cache))

    def _set_capacity(self, engine, features):
        n = features.shape[1]
        if self.config.total_budget_bytes is None:
            self.capacity = self.config.long_pairs
            self.budget_details = {'mode':'legacy-pair-limit'}
            return
        prefix_bytes = 2*engine.system_cache.get_seq_length()*self.kv_bytes_per_token
        # Charge both frozen visual features and all three int64 MRoPE axes.
        short_bytes = self.config.short_pairs*n*(features.shape[-1]*features.element_size()+3*8)
        fixed = (prefix_bytes+short_bytes+self.config.decoded_frame_reserve_bytes+
                 self.config.budget_metadata_reserve_bytes)
        pair_bytes=n*self.kv_bytes_per_token
        pair_metadata=192  # tuple, two timestamps, list slot and safety margin
        self.capacity = (self.config.total_budget_bytes-fixed)//(pair_bytes+pair_metadata)
        if self.capacity < 1:
            raise MemoryError('DSCache budget cannot hold prefix, recent features, reserved frame, and one historical pair')
        self.budget_details = {'mode':'total-persistent-plus-extra-static-bytes',
                              'limit_bytes':self.config.total_budget_bytes,
                              'prefix_copies_bytes':prefix_bytes,'maximum_recent_features_and_positions_bytes':short_bytes,
                              'decoded_frame_reserve_bytes':self.config.decoded_frame_reserve_bytes,
                              'metadata_and_static_reserve_bytes':self.config.budget_metadata_reserve_bytes,
                              'historical_pair_tensor_bytes':pair_bytes,
                              'historical_pair_metadata_reserve_bytes':pair_metadata}

    def _time_span(self, engine, positions):
        if engine.pending:
            return (float(engine.pending[0][0]),float(engine.pending[-1][0]))
        start=float(positions[0,0,0]-engine.video_origin)/getattr(engine.model.config.vision_config,'tokens_per_second',2)
        return (start,start)

    def ingest(self, engine, features, positions, grid):
        if self.tokens_per_pair not in (None, features.shape[1]):
            raise ValueError('DSCache requires a constant spatial grid within a video')
        self.tokens_per_pair = features.shape[1]
        self._set_capacity(engine, features)
        self.short.append((features.detach(),positions.detach(),self._time_span(engine,positions)))
        while len(self.short) > self.config.short_pairs:
            feats,pos,span = self.short.pop(0)
            # One-pair transient overflow is a workspace, not a retained archive.
            self.cache = engine.forward(feats,self.cache,pos).past_key_values
            self.long_times.append(span)
            excess = len(self.long_times)-self.capacity
            if excess > 0:
                self._evict(engine,excess)

    def _evict(self, engine, count):
        prefix = engine.system_cache.get_seq_length()
        remove = count*self.tokens_per_pair
        self.cache = cache_from([(torch.cat((k[:,:,:prefix],k[:,:,prefix+remove:]),2),
                                 torch.cat((v[:,:,:prefix],v[:,:,prefix+remove:]),2))
                                for k,v in pairs(self.cache)])
        del self.long_times[:count]
        self.evictions += count

    def query_cache(self, engine, embeddings):
        # Isolated copy: current question KV can never enter persistent history.
        cache = clone_cache(self.cache)
        for features,positions,_ in self.short:
            cache = engine.forward(features,cache,positions).past_key_values
        return cache

    def stats(self, engine):
        spans = self.long_times+[x[2] for x in self.short]
        return {'long_capacity_pairs':self.capacity,'long_pairs':len(self.long_times),
                'short_pairs':len(self.short),'tokens_per_pair':self.tokens_per_pair,
                'long_visual_tokens':len(self.long_times)*(self.tokens_per_pair or 0),
                'short_feature_tokens':sum(x[0].shape[1] for x in self.short),
                'evicted_pairs':self.evictions,
                'oldest_retained_timestamp':spans[0][0] if spans else None,
                'newest_retained_timestamp':spans[-1][1] if spans else None,
                'retained_time_span_seconds':spans[-1][1]-spans[0][0] if spans else 0.,
                'kv_bytes_per_token':self.kv_bytes_per_token,'budget':dict(self.budget_details)}

    def state(self):
        return {'long_cache':self.cache,'short_visual_features_and_positions':self.short,
                'long_pair_timestamps':self.long_times,'evicted_pairs':self.evictions,
                'tokens_per_pair':self.tokens_per_pair,'long_capacity_pairs':self.capacity,
                'budget':self.budget_details}
