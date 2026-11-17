"""Write compact, auditable GPU smoke evidence without publishing raw traces."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path)
    p.add_argument('--cpu-log',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    root=Path(__file__).parent;run=args.run_dir
    predictions=rows(run/'predictions.jsonl');videos=rows(run/'video_completion.jsonl')
    manifest=json.loads((run/'manifest.json').read_text());progress=json.loads((run/'progress.json').read_text())
    score=json.loads((run/'scoring_final/report.json').read_text());equiv=json.loads((run/'gpu_equivalence.json').read_text())
    metrics=[x['dscache_metrics'] for x in predictions]+[x['metrics'] for x in videos]
    clean=rows(run/'memory_after_cleanup.jsonl');cpu_log=args.cpu_log.read_text()
    sample_count=sum(x['sampled_frames_consumed'] for x in videos)
    elapsed=progress['elapsed_seconds'];query=[x['query_wall_seconds'] for x in predictions]
    assert progress['phase']=='completed' and progress['failed']==0 and len(predictions)==20 and len(videos)==20
    assert all(x['decoder_eof'] and x['all_sampled_frames_consumed'] for x in videos)
    assert all(x['budget_violation'] is None for x in metrics)
    assert all(x['budget_usage']['maximum_bytes']<=manifest['config']['total_budget_bytes'] for x in metrics)
    assert all(len(x['cpu_affinity'])<=2 for x in metrics)
    assert equiv['passed'] and score['task_macro']['task_count']==17
    assert 'Ran 18 tests' in cpu_log and cpu_log.rstrip().endswith('OK')
    result=dict(timestamp=datetime.datetime.now().astimezone().isoformat(),passed=True,
        scope='Diagnostic GPU acceptance on a frozen, task-covering 20-question subset, not benchmark accuracy or a paired method comparison.',
        model=manifest['config']['model_path'],gpu=4,gpu_uuid=manifest['gpu_environment'],
        cpu_tests=dict(passed=True,count=18,log_sha256=sha(args.cpu_log)),
        full_dataset_prepare=dict(questions=3795,videos=3377,excluded_over_12h=2),
        smoke=dict(questions=len(predictions),videos=len(videos),sampled_frames=sample_count,
                   failed=progress['failed'],invalid_format=score['overall']['invalid_format'],
                   overall_sample_accuracy=score['overall']['accuracy'],task_macro=score['task_macro'],
                   wall_seconds_excluding_model_load=elapsed,sampled_frames_per_second=sample_count/elapsed,
                   query_seconds_mean=statistics.mean(query),query_seconds_max=max(query),
                   budget_bytes=manifest['config']['total_budget_bytes'],
                   stable_budget_checks=sum(x['metrics']['budget_usage']['checks'] for x in videos),
                   maximum_charged_bytes=max(x['budget_usage']['maximum_bytes'] for x in metrics),
                   cuda_peak_allocated_bytes=max(x['cuda_peak_allocated'] for x in metrics),
                   cuda_peak_reserved_bytes=max(x['cuda_peak_reserved'] for x in metrics),
                   maximum_process_rss_bytes=max(x['process']['rss'] for x in metrics),
                   process_swap_bytes=max(x['process']['swap'] for x in metrics),
                   after_cleanup_allocated_min_bytes=min(x['allocated'] for x in clean),
                   after_cleanup_allocated_max_bytes=max(x['allocated'] for x in clean),
                   total_evicted_pairs=sum(x['dscache_state']['evicted_pairs'] for x in videos),
                   post_eviction_queries=sum(x['dscache_state']['evicted_pairs']>0 for x in predictions),
                   maximum_query_evicted_pairs=max(x['dscache_state']['evicted_pairs'] for x in predictions),
                   multi_option_response_count=sum(len(x['dscache_prediction']['score']['predicted'] or [])>1 for x in predictions)),
        dense_equivalence={k:v for k,v in equiv.items() if k not in ('metrics','answer')},
        source_sha256={str(f.relative_to(root.parent)):sha(f) for f in sorted(root.glob('*.py'))},
        run_artifact_sha256={name:sha(run/name) for name in ('manifest.json','annotations.json','predictions.jsonl','video_completion.jsonl','gpu_equivalence.json','scoring_final/report.json')},
        prediction_manifest_source_hashes={k:manifest[k] for k in ('runner_sha256','scorer_sha256','prompt_builder_sha256','official_prompt_sha256')},
        postrun_scoring_revision='Finalized-run accounting was added after prediction completion; responses and exact option-set correctness are unchanged. Original scoring/ and manifest retained; final report is in scoring_final/.',
        local_run_dir=str(run.resolve()))
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','cpu_tests','smoke')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
