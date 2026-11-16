#!/usr/bin/env python3
"""Detached supervisor; persist GPU/process/progress snapshots every 20 minutes."""
import datetime,json,os
from pathlib import Path
import subprocess,sys,time
import psutil
out=Path(sys.argv[1]);command=sys.argv[2:]
gpu_index=os.environ['BIOCACHE_PHYSICAL_GPU']
allowed=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,allowed)
with (out/'run.log').open('a',buffering=1) as log:
    worker=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (out/'worker.pid').write_text(str(worker.pid)+'\n')
    (out/'supervisor.pid').write_text(str(os.getpid())+'\n')
    due=0
    while True:
        code=worker.poll()
        if time.monotonic()>=due or code is not None:
            row={'time':datetime.datetime.now().astimezone().isoformat(),'worker_pid':worker.pid,'returncode':code}
            try:
                process=psutil.Process(worker.pid);mem=process.memory_full_info()
                row['process']={'rss':mem.rss,'pss':getattr(mem,'pss',None),'swap':getattr(mem,'swap',None),'threads':process.num_threads(),'cpu_affinity':process.cpu_affinity()}
            except psutil.Error:pass
            try:row['progress']=json.loads((out/'progress.json').read_text())
            except (OSError,ValueError):pass
            gpu=subprocess.run(['nvidia-smi','--id='+gpu_index,'--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
            row['physical_gpu']=int(gpu_index)
            row['gpu'+gpu_index]=gpu.stdout.strip()
            with (out/'monitor_20min.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            due=time.monotonic()+1200
        if code is not None:
            (out/'exit_status.json').write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n');break
        time.sleep(10)
sys.exit(code)
