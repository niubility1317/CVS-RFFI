"""One-shot N607 read-only resource audit; no remote files are written."""
import collections
import datetime
import heapq
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = '/home/szu2070436088/2510044040/CV-SincNet'

def read(path):
    try:
        return Path(path).read_text()
    except (OSError, UnicodeError):
        return None

def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=12)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except subprocess.TimeoutExpired:
        return dict(status='TIMEOUT')

def snapshot():
    procs = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d/'cmdline').read_bytes().decode(errors='replace').split('\0')[:-1]
            cwd = os.readlink(d/'cwd')
            if not (ROOT in cwd or any(ROOT in a for a in argv)):
                continue
            status = read(d/'status') or ''
            if not any(s.startswith('Uid:') and int(s.split()[1]) == os.getuid() for s in status.splitlines()):
                continue
            fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
            item = dict(pid=int(d.name), ppid=int(fields['PPid']), argv=argv, cwd=cwd,
                rss_kib=int(fields.get('VmRSS','0 kB').split()[0]),
                hwm_kib=int(fields.get('VmHWM','0 kB').split()[0]),
                swap_kib=int(fields.get('VmSwap','0 kB').split()[0]),
                threads=int(fields['Threads']), io=read(d/'io'), stat=read(d/'stat'))
            rollup = read(d/'smaps_rollup') or ''
            item['pss_kib'] = next((int(x.split()[1]) for x in rollup.splitlines() if x.startswith('Pss:')), None)
            item['private_kib'] = sum(int(x.split()[1]) for x in rollup.splitlines() if x.startswith(('Private_Clean:','Private_Dirty:')))
            writable = []
            for fd in (d/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    info = read(d/'fdinfo'/fd.name) or ''
                    flags = next(int(x.split()[1],8) for x in info.splitlines() if x.startswith('flags:'))
                    if flags & 3 and target.startswith('/'):
                        st=fd.stat()
                        writable.append(dict(fd=int(fd.name), path=target, bytes=st.st_size,
                            allocated_bytes=st.st_blocks*512, deleted=target.endswith(' (deleted)')))
                except (OSError, StopIteration):
                    continue
            item['writable_fds'] = writable
            cg = read(d/'cgroup') or ''
            item['cgroup'] = cg.strip()
            cgpath = cg.split('::',1)[-1].strip()
            limits=[]
            node=Path('/sys/fs/cgroup')/cgpath.lstrip('/')
            while str(node).startswith('/sys/fs/cgroup'):
                entry=dict(path=str(node))
                for name in ('memory.current','memory.max','memory.high','memory.events','memory.pressure'):
                    entry[name]=read(node/name)
                limits.append(entry)
                if node == Path('/sys/fs/cgroup'): break
                node=node.parent
            item['cgroup_ancestors']=limits
            procs.append(item)
        except (OSError, ValueError, KeyError):
            continue
    return dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        identity=command(['id','-un']), host=command(['hostname']),
        meminfo=read('/proc/meminfo'), memory_pressure=read('/proc/pressure/memory'),
        io_pressure=read('/proc/pressure/io'), vmstat=read('/proc/vmstat'),
        cpus=os.cpu_count(), loadavg=read('/proc/loadavg'), processes=procs,
        disk=command(['df','-B1','/home','/tmp','/dev/shm']),
        gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits']),
        gpu_processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader,nounits']))

def inventory(scope=ROOT):
    os.nice(10)
    start=time.monotonic()
    excluded={'Dataset_ORALCE','Dataset_WigSig','datasets','stage2_inputs','.git','__pycache__'}
    totals=collections.defaultdict(lambda:dict(files=0,bytes=0,allocated_bytes=0))
    by_ext=collections.defaultdict(lambda:dict(files=0,bytes=0,allocated_bytes=0))
    leaders=collections.defaultdict(list)
    errors=[]; skipped=[]; count=0; complete=True
    for folder,dirs,files in os.walk(scope,followlinks=False,onerror=lambda e:errors.append(str(e))):
        for name in dirs[:]:
            path=Path(folder)/name
            if name in excluded or path.is_symlink():
                skipped.append(str(path));dirs.remove(name)
        for name in files:
            if count >= 1000000 or time.monotonic()-start > 45:
                complete=False;break
            path=Path(folder)/name
            try:
                if path.is_symlink():continue
                st=path.stat();count+=1
                rel=path.relative_to(ROOT)
                group=str(Path(*rel.parts[:2])) if len(rel.parts)>2 else str(rel.parts[0])
                ext=path.suffix.lower() or '<none>'
                for aggregate in (totals[group], by_ext[ext]):
                    aggregate['files']+=1;aggregate['bytes']+=st.st_size;aggregate['allocated_bytes']+=st.st_blocks*512
                types=['all']
                if ext in {'.log','.out','.jsonl','.csv','.txt'}:types.append('logs_and_text')
                if ext in {'.log','.out'}:types.append('console_logs')
                if ext=='.jsonl':types.append('jsonl')
                if ext=='.json' and st.st_size>10*1024*1024:types.append('large_json')
                if ext in {'.pth','.pt','.ckpt'}:types.append('checkpoints')
                for kind in types:
                    entry=(st.st_size,str(rel),st.st_blocks*512,st.st_mtime)
                    heapq.heappush(leaders[kind],entry)
                    if len(leaders[kind])>30:heapq.heappop(leaders[kind])
            except OSError as e:
                errors.append(str(e))
        if not complete:break
    return dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope=scope, excluded_directory_names=sorted(excluded), skipped=skipped,
        complete=complete, file_count=count, seconds=time.monotonic()-start, errors=errors,
        totals=dict(sorted(totals.items(),key=lambda x:x[1]['bytes'],reverse=True)),
        extensions=dict(sorted(by_ext.items(),key=lambda x:x[1]['bytes'],reverse=True)),
        largest={k:[dict(bytes=e[0],path=e[1],allocated_bytes=e[2],mtime=e[3]) for e in sorted(v,reverse=True)] for k,v in leaders.items()})

def inventory_runs():
    return inventory(ROOT+'/runs')

def inventory_releases():
    return inventory(ROOT+'/releases')

def live_logs():
    run=Path(ROOT)/'runs/20261001-phase1-cvs-simplex-identity-manysig-m8-r01'
    rows=[]
    for folder in sorted(run.glob('*/source')):
        item=dict(row=folder.parent.name,files=[])
        for name in ('step_metrics.jsonl','epoch_metrics.jsonl','epoch_metrics.csv'):
            path=folder/name
            if not path.exists():continue
            st=path.stat(); lines=0; max_bytes=0; keys=set(); large_arrays=[]; last=None; invalid=0; duplicated=0; seen=set()
            with path.open('rb') as stream:
                for line in stream:
                    if not line.endswith(b'\n'):continue
                    lines+=1;max_bytes=max(max_bytes,len(line))
                    if name.endswith('.jsonl'):
                        try:rec=json.loads(line)
                        except ValueError:invalid+=1;continue
                        keys.update(rec)
                        identity=rec.get('step') if name.startswith('step') else rec.get('epoch')
                        if identity in seen:duplicated+=1
                        seen.add(identity)
                        last={k:rec.get(k) for k in ('epoch','step','optimizer_steps_total','elapsed_seconds','peak_cuda_allocated_bytes')}
                        for k,v in rec.items():
                            if isinstance(v,list) and len(v)>32:large_arrays.append(dict(key=k,length=len(v)))
            item['files'].append(dict(name=name,bytes=st.st_size,lines=lines,max_line_bytes=max_bytes,
                schema_keys=sorted(keys),last_resource_record=last,invalid_complete_lines=invalid,
                duplicate_step_or_epoch=duplicated,large_arrays=large_arrays[:10]))
        config=json.loads((folder/'resolved_config.json').read_text())
        item['resolved']={k:config.get(k) for k in ('pid','commit','steps_per_epoch','epochs','batch_size','precision','total_parameters','trainable_parameters')}
        rows.append(item)
    sizes=command(['nice','-n','10','du','-x','-B1','-s',ROOT+'/logs',ROOT+'/runs',ROOT+'/releases'])
    return dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,directory_sizes=sizes,
        dataset_bytes=Path(ROOT+'/Dataset_WigSig/ManySig.pkl').stat().st_size)

def directory_sizes():
    result={}
    for name in ('project',):
        p=subprocess.run(['nice','-n','10','du','-x','-B1','--max-depth=1',ROOT],
            capture_output=True,text=True,timeout=45)
        rows=[]
        for line in p.stdout.splitlines():
            size,path=line.split('\t',1); rows.append(dict(bytes=int(size),path=path))
        result[name]=dict(returncode=p.returncode,stderr=p.stderr,
            directories=sorted(rows,key=lambda x:x['bytes'],reverse=True),complete=p.returncode==0)
    return dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sizes=result,
        inodes=command(['df','-i','/home','/tmp']))

def focused_d92():
    selected=[
        '20261001-phase2-d92-affine-joint-support-m2-r01',
        '20261001-phase2-d92-margin-joint-support-m2-r01',
        '20261001-phase2-d92-anchor-joint-support-m2-r01',
        '20261001-phase2-d92-conditional-joint-support-m2-r01',
        '20261001-phase2-d92-group-barrier-joint-support-m2-r01',
        '20261001-phase2-d92-margin-joint-repeat-m2-r01',
        '20260929-phase2-d92-branch-local-margin-support-m4-r01']
    results=[]
    for run in selected:
        root=Path(ROOT)/'runs'/run;names=collections.defaultdict(lambda:dict(files=0,bytes=0,allocated_bytes=0))
        extensions=collections.defaultdict(lambda:dict(files=0,bytes=0,allocated_bytes=0));errors=[]
        if not root.is_dir():
            results.append(dict(run=run,status='ABSENT'));continue
        for folder,dirs,files in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
            dirs[:]=[d for d in dirs if not (Path(folder)/d).is_symlink()]
            for name in files:
                path=Path(folder)/name
                if path.is_symlink():continue
                try:st=path.stat()
                except OSError as e:errors.append(str(e));continue
                for agg in (names[name],extensions[path.suffix]):
                    agg['files']+=1;agg['bytes']+=st.st_size;agg['allocated_bytes']+=st.st_blocks*512
        results.append(dict(run=run,status='VERIFIED' if not errors else 'PARTIAL',errors=errors,
            files=sum(x['files'] for x in names.values()),bytes=sum(x['bytes'] for x in names.values()),
            allocated_bytes=sum(x['allocated_bytes'] for x in names.values()),
            by_name=dict(sorted(names.items(),key=lambda x:x[1]['bytes'],reverse=True)[:25]),
            by_extension=dict(sorted(extensions.items(),key=lambda x:x[1]['bytes'],reverse=True))))
    return dict(observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=results,
        content_reads=False,scope='Seven named D92 run trees, metadata only')

def local_main():
    mode=sys.argv[1]
    destination=Path(sys.argv[2])
    source=Path(__file__).read_bytes().replace(b'\r\n',b'\n')
    compile(source, str(__file__), 'exec')
    payload=source+b'\nprint(json.dumps('+mode.encode()+b'(),ensure_ascii=False))\n'
    args=['C:/Windows/System32/OpenSSH/ssh.exe','-F','E:/type10-7/tools/n607_ssh_config',
        '-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -']
    p=subprocess.run(args,input=payload,capture_output=True,timeout=65)
    if p.returncode:
        print(p.stderr.decode(errors='replace'));raise SystemExit(p.returncode)
    result=json.loads(p.stdout.decode('utf-8'))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    check=json.loads(destination.read_text(encoding='utf-8'))
    print(json.dumps(dict(status='VERIFIED',path=str(destination),mode=mode,
        observed_at=check['observed_at'],process_count=len(check.get('processes',[])),
        file_count=check.get('file_count'),complete=check.get('complete')),ensure_ascii=False))

if __name__=='__main__' and os.name=='nt':local_main()
