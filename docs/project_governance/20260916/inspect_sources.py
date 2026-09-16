from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import subprocess
import csv

ROOT = Path('E:/type10-7')
OUT = ROOT / 'docs/project_governance/20260916'
OUT.mkdir(parents=True, exist_ok=True)
repo = ROOT / 'github_publish/CVS-RFFI-repo'
text = subprocess.check_output(['git','-C',str(repo),'worktree','list','--porcelain'],text=True,encoding='utf-8')
worktrees = []
for block in text.strip().split('\n\n'):
    row = dict(line.split(' ', 1) if ' ' in line else (line, True) for line in block.splitlines())
    worktrees.append(row)

def status(row):
    path = Path(row['worktree'])
    if not path.exists():
        return {**row, 'status': 'MISSING_NO_ACTION'}
    r = subprocess.run(['git','-C',str(path),'status','--porcelain','--untracked-files=normal'],capture_output=True,timeout=90)
    lines = r.stdout.decode('utf-8',errors='replace').splitlines()
    return {**row,'status':'DIRTY_KEEP' if lines else 'CLEAN_KEEP_UNREVIEWED_LINEAGE',
            'status_exit':r.returncode,'tracked_changes':sum(not s.startswith('??') for s in lines),
            'untracked_entries':sum(s.startswith('??') for s in lines),'status_sample':lines[:8]}

with ThreadPoolExecutor(max_workers=4) as pool:
    result=list(pool.map(status,worktrees))
(OUT/'git_worktrees.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

rows=[]
for base in ['code','tools','tests','baselines','Fedbase','paper_reproduction']:
    path=ROOT/base
    if not path.exists(): continue
    for p in path.rglob('*.py'):
        rel=p.relative_to(ROOT)
        if any(x in rel.parts for x in ['snapshots','__pycache__','.git']) or any(x.startswith('.pytest') for x in rel.parts): continue
        target=repo/rel
        if not target.is_file():
            verdict='LOCAL_ONLY_KEEP'
        else:
            a,b=p.read_bytes(),target.read_bytes()
            verdict='IDENTICAL_ROLE_MIRROR_KEEP' if a==b else ('EOL_ONLY_ROLE_MIRROR_KEEP' if a.replace(b'\r\n',b'\n')==b.replace(b'\r\n',b'\n') else 'DIFFERENT_VARIANTS_KEEP')
        rows.append({'workspace_path':rel.as_posix(),'git_mirror':target.relative_to(ROOT).as_posix() if target.exists() else '',
                     'status':verdict,'bytes':p.stat().st_size})
with (OUT/'code_ownership.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['workspace_path','git_mirror','status','bytes'],lineterminator='\n'); w.writeheader(); w.writerows(rows)
from collections import Counter
print(json.dumps({'worktrees':len(result),'worktree_status':dict(Counter(x['status'] for x in result)),
                  'source_pairs':len(rows),'source_status':dict(Counter(x['status'] for x in rows))},ensure_ascii=False))
