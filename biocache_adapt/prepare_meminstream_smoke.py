"""Freeze a diagnostic 20-question selection before running the model."""
import collections
import hashlib
import json
from pathlib import Path


def main():
    base=Path('/data/user25101366/Benchmark-for-BioCache/MemInStream/data/MemInStream')
    rows=json.loads((base/'annotations.json').read_text())
    meta={x['file']:x for x in json.loads((base/'statistics/video_metadata.json').read_text())}
    by_video=collections.defaultdict(list)
    for i,row in enumerate(rows):
        if meta[Path(row['video_path']).name]['duration']<=43200:
            by_video[row['video_path']].append(i)
    selected=[];paths=set()
    tasks={(rows[i]['level1'],rows[i]['level2']) for ids in by_video.values() for i in ids}
    uncovered=set(tasks)
    while uncovered:
        def key(path):
            covered=uncovered & {(rows[i]['level1'],rows[i]['level2']) for i in by_video[path]}
            duration=meta[Path(path).name]['duration']
            return (duration/len(covered),duration,path) if covered else (float('inf'),0,path)
        path=min(by_video,key=key);paths.add(path)
        for i in sorted(by_video[path],key=lambda i:(float(rows[i]['query_time']),i)):
            task=(rows[i]['level1'],rows[i]['level2'])
            if task in uncovered:selected.append(i);uncovered.remove(task)
    def add(candidates):
        i=min(candidates,key=lambda i:(rows[i]['video_path'] not in paths,
                                      meta[Path(rows[i]['video_path']).name]['duration'],i))
        if i not in selected:selected.append(i);paths.add(rows[i]['video_path'])
    add([i for ids in by_video.values() for i in ids if isinstance(rows[i].get('correct_index'),list)
         and len(rows[i]['correct_index'])>1])
    long=[]
    for path,ids in by_video.items():
        m=meta[Path(path).name];w,h=m['width'],m['height'];scale=min(1.,(153664/(w*h))**.5)
        n=max(28,int(w*scale)//28*28)*max(28,int(h*scale)//28*28)//784
        if n>=168:long.extend(i for i in ids if float(rows[i]['query_time'])>400)
    add(long)
    while len(selected)<20:
        add([i for ids in by_video.values() for i in ids if i not in selected])
    assert len(selected)==20
    selected.sort();out=Path(__file__).parent/'fixtures';out.mkdir(exist_ok=True)
    fixture=out/'meminstream_smoke20_indices.json';fixture.write_text(json.dumps(selected,indent=2)+'\n')
    summary=dict(selection_policy='Minimum-duration video coverage of 17 level2 tasks; add genuine multi-answer and query after 400s at >=168 visual tokens/pair; fill 20 preferring selected videos. Frozen before predictions; diagnostic only.',
        annotation_sha256=hashlib.sha256((base/'annotations.json').read_bytes()).hexdigest(),
        selection_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),questions=len(selected),videos=len(paths),
        source_duration_seconds=sum(meta[Path(p).name]['duration'] for p in paths),task_count=len(tasks),
        selected=[dict(original_index=i,level1=rows[i]['level1'],level2=rows[i]['level2'],
                       query_time=rows[i]['query_time'],video=rows[i]['video_path']) for i in selected])
    (out/'meminstream_smoke20_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='selected'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
