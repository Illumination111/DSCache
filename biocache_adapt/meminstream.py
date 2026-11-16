#!/usr/bin/env python3
"""Causal native-Qwen DSCache evaluation; one model, one video at a time."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='2'
os.environ['TOKENIZERS_PARALLELISM']='false'
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from .meminstream_prompts import build_request, is_multiselect

ROOT=Path(__file__).resolve().parents[1]

from .qwen_stream import QwenStreamAdapter, StreamConfig, limit_cpu
limit_cpu(2)
import av
import torch


def now(): return datetime.datetime.now().astimezone().isoformat()
def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def frames(path, decode_stats=None):
    """Latest visible frame at each 1-second grid; return its actual relative PTS."""
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_count=2
        origin=float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.
        previous=None;grid=0.;last=-1.
        for frame in container.decode(stream):
            if decode_stats is not None:
                decode_stats["decoded_source_frames"] = decode_stats.get("decoded_source_frames", 0) + 1
            if frame.pts is None: raise ValueError('video frame has no PTS')
            stamp=float(frame.pts*stream.time_base)-origin
            if stamp<0:continue
            while previous is not None and grid<stamp-1e-8:
                if previous[0]>last:
                    last=previous[0];yield previous
                grid+=1.
            previous=(stamp,frame)
        if previous is not None and grid<=previous[0]+1e-8 and previous[0]>last:
            yield previous


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--data-root',default='/data/user25101366/Benchmark-for-BioCache/MemInStream/data/MemInStream')
    p.add_argument('--budget-gib',type=float,choices=(0.875,1.3125,1.75),default=1.75)
    p.add_argument('--short-pairs',type=int,default=4)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return run(args,out)


def run(args,out):
    if args.resume and not (out/'manifest.json').exists():
        raise FileNotFoundError('resume requires an existing manifest')
    if not args.resume and ((out/'manifest.json').exists() or (out/'predictions.jsonl').exists()):
        raise FileExistsError('output already contains a run; use --resume or a fresh directory')
    data=Path(args.data_root);source=data/'annotations.json';rows=json.loads(source.read_text())
    meta=json.loads((data/'statistics/video_metadata.json').read_text())
    excluded={x['file'] for x in meta if x['duration']>43200}
    selected=[(i,r) for i,r in enumerate(rows) if Path(r['video_path']).name not in excluded]
    filtered=[r for _,r in selected]
    groups={}
    for index,(original,row) in enumerate(selected):
        groups.setdefault(row['video_path'],[]).append((index,original,row))
    for path in groups:
        if not (data/path).is_file():raise FileNotFoundError(data/path)
    config=StreamConfig(device=args.device,total_budget_bytes=round(args.budget_gib*1024**3),short_pairs=args.short_pairs)
    video_meta={x["file"]:x for x in meta}
    manifest={'created_at':now(),'method':'DSCache','python':sys.executable,
              'cache_budget_semantics':'total persistent plus extra static CPU/GPU bytes; adaptive FIFO historical capacity; recent features and raw frame reserve charged',
              'config':config.__dict__,'excluded_videos':sorted(excluded),'questions':len(filtered),'videos':len(groups),
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'video_metadata_sha256':hashlib.sha256((data/'statistics/video_metadata.json').read_bytes()).hexdigest(),
              'sampling':'1 Hz grid, latest visible frame, actual relative video PTS',
              'generation':'greedy, native EOS, 256 max new tokens',
              'full_video_policy':'consume to decoder EOF after causal queries; all sampled pairs enter recent feature buffer; only expired recent pairs enter FIFO long KV; odd tail repeats last visible frame',
              'options_prompt':'v2: single-choice returns exactly one letter; normalized multi-select returns all correct letters; no label values/count in model request',
              'prompt_builder_sha256':hashlib.sha256((ROOT/'biocache_adapt/meminstream_prompts.py').read_bytes()).hexdigest(),
              'question_mode_counts':{'single':sum(not is_multiselect(r) for r in filtered),'multiple':sum(is_multiselect(r) for r in filtered)},
              'dialogue':'only current sample supplied dialogue events visible by query time',
              'gpu_environment':os.environ.get('CUDA_VISIBLE_DEVICES'),
              'adapter_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (ROOT/'biocache_adapt').glob('*.py')},
              'policy_sha256':hashlib.sha256((ROOT/'biocache_adapt/qwen_policy.py').read_bytes()).hexdigest(),
              'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scorer_sha256':hashlib.sha256((ROOT/'biocache_adapt/meminstream_scoring.py').read_bytes()).hexdigest(),
              'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'torch':torch.__version__,'av':av.__version__}
    manifest=json.loads(json.dumps(manifest))
    if args.resume and (out/'manifest.json').exists():
        old=json.loads((out/'manifest.json').read_text())
        for key in ['source_sha256','video_metadata_sha256','config','sampling','options_prompt','full_video_policy']:
            if old.get(key)!=manifest.get(key):raise ValueError(f'resume fingerprint mismatch: {key}')
        with (out/'execution_revisions.jsonl').open('a') as f:f.write(json.dumps(manifest,ensure_ascii=False)+'\n')
    else:save(out/'manifest.json',manifest)
    if args.resume and (out/'annotations.json').exists() and json.loads((out/'annotations.json').read_text())!=filtered:
        raise ValueError('saved filtered annotations differ from current input')
    if not (out/'annotations.json').exists():save(out/'annotations.json',filtered)
    if args.prepare_only:
        print(json.dumps({'questions':len(filtered),'videos':len(groups),'excluded':len(excluded)}));return
    existing={};done_videos=set();previous_elapsed=0.
    if (out/'predictions.jsonl').exists():
        if not args.resume:raise RuntimeError('use --resume for existing predictions')
        raw=(out/'predictions.jsonl').read_bytes()
        if raw and not raw.endswith(b'\n'):
            valid,_,tail=raw.rpartition(b'\n')
            (out/'incomplete_prediction.fragment').write_bytes(tail)
            (out/'predictions.jsonl').write_bytes(valid+b'\n' if valid else b'')
        for line in (out/'predictions.jsonl').read_text().splitlines():
            entry=json.loads(line);index=entry['dscache_prediction']['input_index']
            if type(index) is not int or not 0<=index<len(filtered):raise ValueError('invalid saved prediction index')
            if index in existing:raise ValueError('duplicate saved prediction')
            if {k:v for k,v in entry.items() if k in filtered[index]}!=filtered[index]:raise ValueError('saved prediction/input mismatch')
            existing[index]=entry
        if (out/'video_completion.jsonl').exists():
            done_videos={json.loads(line)['video'] for line in (out/'video_completion.jsonl').read_text().splitlines() if line.strip()}
        if not done_videos.issubset(groups):raise ValueError('checkpoint contains unknown videos')
        if any(index not in existing for path in done_videos for index,_,_ in groups[path]):
            raise ValueError('completed video has missing saved answers')
        if (out/'progress.json').exists():previous_elapsed=json.loads((out/'progress.json').read_text()).get('elapsed_seconds',0.)
    progress={'started_at':now(),'phase':'loading_model','completed':0,'failed':0,'total':len(filtered),'videos_total':len(groups),'videos_completed':0}
    if args.resume and (out/'progress.json').exists():
        progress['started_at']=json.loads((out/'progress.json').read_text())['started_at']
    progress.update(completed=len(existing),videos_completed=len(done_videos),resumed_at=now() if args.resume else None)
    save(out/'progress.json',progress)
    try:
        model=QwenStreamAdapter.load(config)
    except BaseException as exc:
        progress.update(phase='failed',error_type=type(exc).__name__,error=str(exc),finished_at=now());save(out/'progress.json',progress);raise
    evaluator_path=ROOT/'biocache_adapt/meminstream_scoring.py'
    spec=importlib.util.spec_from_file_location('offline_scoring',evaluator_path);scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    scores=[r['dscache_prediction']['score']['correct'] for r in existing.values()]
    scored=sum(type(v) is bool for v in scores);correct=sum(v is True for v in scores);start=time.time()-previous_elapsed
    try:
        with (out/'predictions.jsonl').open('a',buffering=1) as predictions,(out/'failures.jsonl').open('a',buffering=1) as failures, (out/'video_completion.jsonl').open('a',buffering=1) as completions:
            for path,questions in groups.items():
                if path in done_videos:continue
                questions.sort(key=lambda x:(float(x[2]['query_time']),x[0]))
                progress.update(phase='video_prefill',video=path,video_index=progress['videos_completed']+1)
                save(out/'progress.json',progress)
                pending={x[0] for x in questions if x[0] not in existing};decoder=None
                try:
                    vm=video_meta[Path(path).name]
                    model.config.decoded_frame_reserve_bytes=vm['width']*vm['height']*3
                    model.open_stream(Path(path).stem)
                    decode_stats={}
                    decoder=frames(data/path,decode_stats);lookahead=next(decoder,None)
                    for index,original,row in questions:
                        query_time=float(row['query_time'])
                        while lookahead is not None and lookahead[0]<=query_time+1e-8:
                            stamp,frame=lookahead
                            model.push_frames([(stamp,frame.to_ndarray(format='rgb24'))],visible_until=query_time+1e-8)
                            if model.frames_seen%10==0:
                                progress.update(phase='video_prefill',frame_time=stamp,frames_seen=model.frames_seen);save(out/'progress.json',progress)
                            lookahead=next(decoder,None)
                        if index in existing:continue
                        model.dialogue_events=[]
                        for event in sorted(row.get('dialogue_events',[]),key=lambda e:float(e['time'])):
                            if float(event['time'])<=query_time:
                                model.inject_dialogue(float(event['time']),event['role'],event['content'])
                        options=[re.sub(r'^[A-Z][.)]\s*','',str(x)) for x in (row.get('options') or [])]
                        progress.update(phase='answer',query_index=index,query_time=query_time);save(out/'progress.json',progress)
                        request=build_request(row)
                        response=model.answer(request)
                        score=scorer.score(row,response,'skip')
                        if type(score['correct']) is bool:
                            scored+=1;correct+=int(score['correct'])
                        record=dict(row)
                        record['dscache_prediction']={'input_index':index,'original_input_index':original,'status':'completed','response':response,'score':score,'completed_at':now()}
                        record['prompt_mode']='multiple' if request['multiple'] else 'single'
                        record['dscache_metrics']=model.report()
                        record['dscache_state']=model.policy.stats(model)
                        predictions.write(json.dumps(record,ensure_ascii=False)+'\n')
                        pending.remove(index);progress['completed']+=1
                        progress.update(scored=scored,correct=correct,accuracy_scored=correct/scored if scored else None,elapsed_seconds=time.time()-start)
                        save(out/'progress.json',progress);print(json.dumps(progress,ensure_ascii=False),flush=True)
                    progress.update(phase='video_tail_prefill');save(out/'progress.json',progress)
                    while lookahead is not None:
                        stamp,frame=lookahead
                        model.push_frames([(stamp,frame.to_ndarray(format='rgb24'))])
                        if model.frames_seen%10==0:
                            progress.update(frame_time=stamp,frames_seen=model.frames_seen);save(out/'progress.json',progress)
                        lookahead=next(decoder,None)
                    repeated_tail = bool(model.pending)
                    if model.pending:
                        features,positions,grid=model._prepare_pair([model.pending[0],model.pending[0]])
                        model.policy.ingest(model,features,positions,grid)
                        model.next_position=max(model.next_position,int(positions.max())+1)
                        model.pending.clear()
                    assert not model.pending
                    completions.write(json.dumps({'video':path,'completed_at':now(),
                        'decoder_eof':True,'sampled_frames_consumed':model.frames_seen,
                        'last_sample_timestamp':model.last_timestamp,'odd_tail_repeated':repeated_tail,
                        'decoded_source_frames':decode_stats.get('decoded_source_frames',0),
                        'all_sampled_frames_consumed':True,'recent_features_build_kv_on_demand':True,
                        'dscache_state':model.policy.stats(model),
                        'metrics':model.report()},ensure_ascii=False)+'\n')

                except Exception as exc:
                    for index,original,row in questions:
                        if index in pending:
                            failures.write(json.dumps({'input_index':index,'original_input_index':original,'video':path,'error_type':type(exc).__name__,'error':str(exc),'time':now()},ensure_ascii=False)+'\n');progress['failed']+=1
                    print(f'VIDEO_FAILED {path}: {type(exc).__name__}: {exc}',flush=True)
                    if isinstance(exc,torch.cuda.OutOfMemoryError) or 'CUDA' in str(exc):raise
                finally:
                    if decoder is not None:decoder.close()
                    if model.stream_id is not None:model.close_stream()
                    if model.device.type=='cuda':
                        with (out/'memory_after_cleanup.jsonl').open('a') as f:
                            f.write(json.dumps({'time':now(),'video':path,'allocated':torch.cuda.memory_allocated(model.device),
                                                'reserved':torch.cuda.memory_reserved(model.device)},ensure_ascii=False)+'\n')
                progress['videos_completed']+=1;save(out/'progress.json',progress)
        progress.update(phase='completed',finished_at=now());save(out/'progress.json',progress)
        subprocess.run([sys.executable,str(evaluator_path),'--input',str(out/'annotations.json'),'--predictions',str(out/'predictions.jsonl'),'--output-dir',str(out/'scoring')],check=True)
    except BaseException as exc:
        progress.update(phase='failed',error_type=type(exc).__name__,error=str(exc),finished_at=now());save(out/'progress.json',progress);raise
    finally:model.dispose()

if __name__=='__main__':main()
