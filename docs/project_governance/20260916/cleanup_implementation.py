"""Bounded cleanup for the user's 20260916 request; no recursive removal."""
from pathlib import Path
import sqlite3, json, hashlib, gzip, shutil, zipfile, time

ROOT=Path('E:/type10-7').resolve()
OUT=ROOT/'docs/project_governance/20260916'
STORE=ROOT/'local_artifacts/project_file_management/20260916'
db=sqlite3.connect(STORE/'inventory.sqlite')
boundaries=[ROOT/p for (p,) in db.execute('SELECT path FROM boundaries WHERE kind=?',('git_internal',))]
worktrees=json.loads((OUT/'git_worktrees.json').read_text(encoding='utf-8'))
protected=[Path(x['worktree']).resolve() for x in worktrees]+[p.parent for p in boundaries if p.parent != ROOT]

def safe(p):
    p=p.absolute()
    if not p.is_relative_to(ROOT) or p.resolve()!=p: raise ValueError(str(p))
    for parent in [p,*p.parents]:
        if parent==ROOT: break
        if parent.lstat().st_file_attributes & 0x400: raise ValueError('reparse '+str(parent))
    return p

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

rows=[]
for rel,size,mtime,category in db.execute("SELECT path,bytes,mtime,category FROM files WHERE category IN ('regenerable_python_cache','regenerable_pytest_cache')"):
    p=ROOT/rel
    if rel.split('/')[0] not in {'code','tools','tests','.pytest_cache','__pycache__'}: continue
    if any(p.is_relative_to(w) for w in protected): continue
    if 'snapshots' in p.parts: continue
    # Only genuine bytecode and standard pytest cache files; never arbitrary __pycache__ contents.
    if category=='regenerable_python_cache' and p.suffix not in {'.pyc','.pyo'}: continue
    if category=='regenerable_pytest_cache' and p.name not in {'README.md','.gitignore','CACHEDIR.TAG','nodeids','lastfailed','stepwise'}: continue
    safe(p)
    if not p.is_file() or p.stat().st_size!=size or p.stat().st_mtime!=mtime: continue
    rows.append({'path':rel,'bytes':size,'sha256':digest(p),'action':'cache_removed_with_recovery_copy'})
archive=STORE/'removed_caches.zip'
if archive.exists(): raise RuntimeError('Already executed; inspect receipt before retry')
manifest=OUT/'cleanup_manifest.json'
manifest.write_text(json.dumps({'state':'PLANNED','archive':str(archive),'files':rows},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for row in rows: z.write(ROOT/row['path'],row['path'])
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    for row in rows:
        assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
        p=safe(ROOT/row['path'])
        assert digest(p)==row['sha256']
        p.unlink()
        assert not p.exists()
compressed=[]
old=ROOT/'.codex_tmp/experiment_management_delivery_20260916/initial_generated_indexes'
for name in ['artifacts.jsonl','facts.jsonl']:
    p=safe(old/name)
    if not p.exists(): continue
    dest=STORE/(name+'.gz')
    before=digest(p); size=p.stat().st_size
    if dest.exists(): raise RuntimeError(str(dest))
    with p.open('rb') as src, gzip.open(dest,'xb',compresslevel=6) as dst: shutil.copyfileobj(src,dst)
    h=hashlib.sha256()
    with gzip.open(dest,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    assert before==h.hexdigest()==digest(p)
    p.unlink()
    assert not p.exists()
    compressed.append({'old_path':str(p),'archive':str(dest),'sha256':before,'bytes':size,'archive_bytes':dest.stat().st_size,'restore':'gzip decompress to old_path'})
receipt={'state':'VERIFIED','cache_count':len(rows),'cache_bytes':sum(x['bytes'] for x in rows),'cache_archive':str(archive),'cache_archive_bytes':archive.stat().st_size,'compressed_generated_indexes':compressed,'files':rows,'boundary':'Only named generated indexes and eligible non-worktree caches; no experiment outputs or source removed. Physical disk allocation not measured.'}
receipt['net_logical_bytes_saved']=receipt['cache_bytes']-archive.stat().st_size+sum(x['bytes']-x['archive_bytes'] for x in compressed)
manifest.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='files'},ensure_ascii=False,indent=2))
