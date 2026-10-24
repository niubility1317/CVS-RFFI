"""Bounded local quarantine and removal of superseded comparison deliverables.

Never touches scientific models, received IQ, remote hosts or Git internals.
All archive members are fully read before any original file is removed.
"""
import argparse
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
import stat
import subprocess
import zipfile

ROOT=Path('E:/type10-7').resolve()
QUARANTINE=Path('F:/AI/exports/STRA-RFFI/quarantine/20261024-obsolete-comparisons-r01')
DOC=ROOT/'docs/project_governance/20261024_comparison_retirement'
KIT=ROOT/'exports/unified_figures/20261024-practical-residual-paper-r03'
ACTIVE=['paper/phase1_experiments_ieee_20261020','experiments/practical_full_2x2',
 'local_artifacts/external_practical_residual_2x2_design_20261023/full_complete_20261024']

def safe(path):
    p=path.resolve(strict=True)
    assert p.is_relative_to(ROOT) and p!=ROOT
    for part in [p,*p.parents]:
        if part==ROOT:break
        assert part.name.lower() not in ('.git','code','worktrees')
        assert not part.is_symlink() and not (getattr(part.lstat(),'st_file_attributes',0)&stat.FILE_ATTRIBUTE_REPARSE_POINT),str(part)
    return p

def targets():
    out=[]
    for p in (ROOT/'exports/unified_figures').iterdir():
        if p.name.startswith(('20261022-pamf-','20261023-pamf-complete-results-')):out.append(p)
    for p in (ROOT/'exports/phase1_external_dg').iterdir():
        if p.name.startswith(('20261020-nonpamf-three-seed-', '20261021-final-T1-DURA-experiments-progress-',
            '20261022-final-T1-DURA-experiments-progress-', '20261023-final-T1-DURA-experiments-',
            '20261023-PAMF-IEEE-full-methods-', '20261022-pamf-full-pl-single-charts-',
            '20261022-pamf-seven-scene-grouped-bars-','20261020-ieee-experiments-only-')):out.append(p)
    for s in ['local_artifacts/phase1_dg_sourcecount6_20261021/final_360_complete',
              'local_artifacts/phase1_dg_sourcecount6_20261021/r04_reproduce_check',
              'paper/phase1_experiments_ieee_20261020/FULL_METHODS_REVISION_20261023.md']:
        p=ROOT/s
        if p.exists():out.append(p)
    return sorted(map(safe,out))

def inventory():
    ts=targets();files=[]
    for t in ts:
        members=sorted(t.rglob('*')) if t.is_dir() else [t]
        for p in members:
            safe(p)
            if not p.is_file():continue
            assert p.suffix.lower() not in ('.pt','.pth','.npy','.npz','.h5','.hdf5'),str(p)
            s=p.stat();files.append(dict(path=p.relative_to(ROOT).as_posix(),size=s.st_size,mtime_ns=s.st_mtime_ns))
    needles=[p.relative_to(ROOT).as_posix() for p in ts]
    # Read only the current paper, current method, and its already frozen input metadata.
    matches=[]
    for a in ACTIVE:
        for p in (ROOT/a).rglob('*'):
            if p.suffix not in ('.py','.tex','.json','.md','.bib','.csv') or not p.is_file():continue
            if any(p==t or p.is_relative_to(t) for t in ts):continue
            text=p.read_text(encoding='utf-8-sig',errors='replace').replace('\\','/')
            for n in needles:
                if n in text:matches.append(dict(file=p.relative_to(ROOT).as_posix(),target=n))
    assert not matches,matches
    # Local workload check; command lines are captured privately and never published.
    ps='Get-CimInstance Win32_Process | Where-Object { $_.Name -match "python|latex|pdftoppm" } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress'
    x=subprocess.run(['C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe','-NoProfile','-Command',ps],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    procs=json.loads(x.stdout or '[]');procs=[procs] if isinstance(procs,dict) else procs
    active=[]
    for p in procs:
        cmd=(p.get('CommandLine') or '').replace('\\','/').lower()
        if any(n.lower() in cmd for n in needles):active.append(p.get('ProcessId'))
    assert not active,active
    result=dict(status='READY_FOR_ARCHIVE',targets=[p.relative_to(ROOT).as_posix() for p in ts],
        file_count=len(files),logical_bytes=sum(x['size'] for x in files),active_dependency_matches=matches,
        active_local_process_matches=active,scope='local superseded mixed figure/export packages and obsolete external aggregate copies only',
        protected=['all code and Git worktrees','all original F/G weights and IQ/cache handles','current practical-residual r14 and 627-row inputs/results',
        'historical valid PAMF mechanism/structure/latency sources','experiment registrations, event histories and recovery maps','remote N607 files and processes'],
        deletion_authorization='2026-10-24 user: package/isolate old erroneous experiment data and records, delete after confirming no use',files=files)
    return result

def run(execute=False):
    DOC.mkdir(parents=True,exist_ok=True)
    if execute and (DOC/'cleanup_manifest.json').exists():raise RuntimeError('Reconcile existing cleanup_manifest before retrying')
    inv=inventory()
    (DOC/'inventory.json').write_text(json.dumps({k:v for k,v in inv.items() if k!='files'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in inv.items() if k not in ('files','targets')},ensure_ascii=False),flush=True)
    if not execute:return
    assert (KIT/'historical_PAMF/manifest.json').exists()
    assert (ROOT/'exports/phase1_external_dg/20261024-practical-residual-full2x2-complete-r14.zip').exists()
    QUARANTINE.mkdir(parents=True,exist_ok=False)
    archive=QUARANTINE/'obsolete_local_comparisons.zip'
    manifest=QUARANTINE/'restore_map.json.gz'
    # Store original compressed packages verbatim; deflate loose raster/data to avoid inflated archives.
    with zipfile.ZipFile(archive,'x',allowZip64=True) as z:
        for i,rec in enumerate(inv['files']):
            p=safe(ROOT/rec['path']);assert p.stat().st_size==rec['size'] and p.stat().st_mtime_ns==rec['mtime_ns']
            compression=zipfile.ZIP_STORED if p.suffix.lower() in ('.zip','.gz','.png','.tiff','.jpg','.pdf') else zipfile.ZIP_DEFLATED
            z.write(p,rec['path'],compress_type=compression)
            rec['archive_crc32']=z.getinfo(rec['path']).CRC
            if i%300==0:print('archived',i,'/',len(inv['files']),flush=True)
    with gzip.open(manifest,'wt',encoding='utf-8') as f:json.dump(inv,f,ensure_ascii=False)
    with zipfile.ZipFile(archive) as z:
        assert len(z.infolist())==len(inv['files'])
        assert z.testzip() is None  # Reads and checks every member in full.
        assert all(z.getinfo(x['path']).file_size==x['size'] and z.getinfo(x['path']).CRC==x['archive_crc32'] for x in inv['files'])
    print('FULL_ARCHIVE_READBACK_VERIFIED',flush=True)
    removed=[]
    # Archive verified. Recheck original file state, then remove only inventoried files.
    for rec in inv['files']:
        p=safe(ROOT/rec['path']);s=p.stat();assert s.st_size==rec['size'] and s.st_mtime_ns==rec['mtime_ns']
        p.unlink();removed.append(rec['path'])
    for rel in sorted(inv['targets'],key=len,reverse=True):
        t=ROOT/rel
        if t.is_dir():
            for p in sorted(t.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if p.is_dir():p.rmdir()
            t.rmdir()
        assert not t.exists(),str(t)
    cleanup={k:v for k,v in inv.items() if k!='files'}
    cleanup.update(status='VERIFIED_ARCHIVED_AND_REMOVED',archive=str(archive),archive_bytes=archive.stat().st_size,
        restore_map=str(manifest),full_member_CRC_readback=True,deleted_file_count=len(removed),
        original_target_absence_verified=True,physical_disk_space_reclaimed='not measured; logical source bytes only',
        historical_traceability='Original experiment records and event histories retained with usage notices; mixed packages are recoverable, not permanently destroyed.')
    (DOC/'cleanup_manifest.json').write_text(json.dumps(cleanup,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (QUARANTINE/'README_恢复说明.md').write_text('''# 已隔离的旧比较资料

旧外部方法的增强设置不符合当前 practical residual 比较要求。混合图包同时含有效 PAMF 证据，因此整包保留供追溯，不能整包标成科学无效，也不能直接用于当前论文。

obsolete_local_comparisons.zip 完整保存原文件相对 E:/type10-7 的路径。restore_map.json.gz 记录每个原路径、大小和归档 CRC。删除原副本前已完整读回全部成员并核验 CRC。

恢复时先读取映射，只提取需要的相对路径到新目录，核对后再放回；禁止直接解压覆盖现有新结果。模型权重、IQ、远端文件和实验登记均未删除。当前论文资料请用 20261024-practical-residual-paper-r03；正式冻结结果用 practical-residual r14。
''',encoding='utf-8')
    print(json.dumps(cleanup,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args();run(a.execute)
