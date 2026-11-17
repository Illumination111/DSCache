"""Real frozen-7B checks: dense-prefill equivalence and query isolation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from .qwen_stream import QwenStreamAdapter, StreamConfig, clone_cache, pairs, storage_report, limit_cpu
from .meminstream_prompts import build_request
import torch
import numpy as np


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-file',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();limit_cpu(2)
    start=time.perf_counter()
    engine=QwenStreamAdapter.load(StreamConfig(device=args.device,total_budget_bytes=round(1.75*1024**3),short_pairs=4))
    engine.open_stream('real-7b-equivalence')
    try:
        # Four actually encoded frames, preserving native temporal/spatial MRoPE.
        for i in range(4):
            frame=np.zeros((392,392,3),dtype=np.uint8);frame[:,:,i%3]=160+i*20
            engine.push_frames([(float(i),frame)])
        dense=clone_cache(engine.system_cache)
        for features,positions,_ in engine.policy.short:
            dense=engine.forward(features,dense,positions).past_key_values
        decoupled=engine.policy.query_cache(engine,None)
        errors=[]
        for (dk,dv),(ck,cv) in zip(pairs(dense),pairs(decoupled)):
            errors.extend([float((dk-ck).abs().max()),float((dv-cv).abs().max())])
        assert len(errors)==56 and max(errors)==0., errors
        ids=engine.processor.tokenizer('What colors are visible?',return_tensors='pt',add_special_tokens=False)['input_ids'].to(engine.device)
        pos=torch.arange(engine.next_position,engine.next_position+ids.shape[-1],device=engine.device).view(1,1,-1).expand(3,1,-1)
        ld=engine.model.lm_head(engine.forward(engine.embedding(ids),dense,pos).last_hidden_state[:,-1])
        lc=engine.model.lm_head(engine.forward(engine.embedding(ids),decoupled,pos).last_hidden_state[:,-1])
        logit_error=float((ld-lc).abs().max());assert logit_error==0.
        assert torch.isfinite(ld).all() and torch.isfinite(lc).all()
        del dense,decoupled,ld,lc
        def snapshots():
            return [digest(t) for kv in pairs(engine.policy.cache) for t in kv]+[digest(t) for f,p,_ in engine.policy.short for t in (f,p)]
        before=snapshots()
        request=build_request(dict(question='Which colors are visible?',query_time=3.,options=['A. Red','B. Blue']))
        a=engine.answer(request);tokens_a=engine.last_answer['token_ids']
        b=engine.answer(request);tokens_b=engine.last_answer['token_ids']
        assert a==b and tokens_a==tokens_b and snapshots()==before
        report=engine.report();limit=engine.config.total_budget_bytes
        engine.config.total_budget_bytes=1
        try:
            engine._check_budget()
        except MemoryError:
            strict_guard=True
        else:
            raise AssertionError('strict budget guard did not fire')
        engine.config.total_budget_bytes=limit;engine.budget_violation=None
        engine.close_stream()
        cleanup=dict(allocated=torch.cuda.memory_allocated(engine.device),reserved=torch.cuda.memory_reserved(engine.device))
        result=dict(passed=True,model=engine.config.model_path,gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    torch=torch.__version__,cpu_affinity=sorted(os.sched_getaffinity(0)),
                    encoded_frames=4,visual_tokens_per_pair=196,dense_kv_max_abs_error=max(errors),
                    dense_next_logits_max_abs_error=logit_error,repeated_answers_identical=True,
                    persistent_tensors_unchanged_after_queries=True,strict_budget_guard_passed=strict_guard,
                    answer=a,elapsed_seconds=time.perf_counter()-start,metrics=report,after_cleanup=cleanup,
                    scope='Exact prefill check before any eviction, within this source-position MRoPE port; not upstream position-agnostic reproduction.')
        args.output_file.parent.mkdir(parents=True,exist_ok=True)
        args.output_file.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('metrics',)},ensure_ascii=False,indent=2))
    finally:
        if engine.stream_id is not None:engine.close_stream()
        engine.dispose()

if __name__=='__main__':main()
