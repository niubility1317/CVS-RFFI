from pathlib import Path
import json,shutil,datetime,subprocess
P=Path(__file__).parent
root=Path('E:/type10-7');reg=root/'automation_reports/CV-SincNet'/P.name
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
read=json.loads((P/'data_readback.json').read_text(encoding='utf-8'))
s=json.loads((reg/'experiment.json').read_text(encoding='utf-8'))
s.update(status='RUNNING',delivery='VERIFIED',updated_at=now)
for row,live in zip(s['rows'],read):
    row.update(status=live['status'],pid=live['pid'],gpu=live['gpu'])
(reg/'experiment.json').write_bytes(json.dumps(s,ensure_ascii=False,indent=2).encode())
report=(P/'report.md').read_text(encoding='utf-8').replace('状态：PREPARED。','状态：RUNNING，远端启动VERIFIED。')
report+='\n## 启动读回\n\n'
for live in read:
    report+=f"- {live['run_id']}：GPU{live['gpu']}，PID{live['pid']}，已记录{live['batches']}批；实际U加载数{live['data_summary']['unlabeled_loaded']}。\n"
report+='\n代码版本：29361aea1554ff69d218a23cafa252af02a8d35e。STAR本地仓库无remote，代码及证据同时镜像至CVS发布仓库。实际PID/CWD/argv见remote_inspect.json，加载数及增强边界见data_readback.json。\n'
(P/'report.md').write_bytes(report.encode());(reg/'report.md').write_bytes(report.encode())
with (reg/'events.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(time=now,status='RUNNING',evidence=str(P/'data_readback.json'),note='Independent PID/CWD/argv/GPU and advancing batches verified'),ensure_ascii=False)+'\n')
old=root/'automation_reports/CV-SincNet/20260920-riei-original-concat-pl-s299-e200-r03'
o=json.loads((old/'experiment.json').read_text(encoding='utf-8'));o.update(status='ANALYZED',updated_at=now)
for row in o['rows']:row['status']='ANALYZED'
(old/'experiment.json').write_bytes(json.dumps(o,ensure_ascii=False,indent=2).encode())
note='\n## 2026-09-20完成复核\n\n已完成200轮/5800批；冻结预测独立重算：source clean88%、source SG22%、target clean57.375%、target SG17.5%。无日志异常。结果仅作已知目标条件下的探索诊断，详见相邻r04的diagnosis.json；旧产物保留。\n'
with (old/'report.md').open('a',encoding='utf-8',newline='\n') as f:f.write(note)
with (old/'events.jsonl').open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(dict(time=now,status='ANALYZED',evidence=str(P/'diagnosis.json'),note='E200 complete and frozen predictions independently rescored'))+'\n')
pub=root/'github_publish/CVS-RFFI-repo';paths=[]
for source in [reg,old]:
    dest=pub/'automation_reports/CV-SincNet'/source.name;dest.mkdir(parents=True,exist_ok=True)
    for f in source.iterdir():
        if f.is_file():shutil.copyfile(f,dest/f.name);paths.append(str((dest/f.name).relative_to(pub)))
dest=pub/'automation_reports/CV-SincNet'/P.name/'release';dest.mkdir(exist_ok=True)
for f in P.iterdir():
    if f.suffix in ['.py','.json','.md'] or f.name=='release_sha256.txt':
        shutil.copyfile(f,dest/f.name);paths.append(str((dest/f.name).relative_to(pub)))
(P/'publish_paths.json').write_bytes(json.dumps(paths).encode())
print('REGISTERED_AND_MIRRORED',len(paths))
