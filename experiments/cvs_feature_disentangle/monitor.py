"""Read-only process, resolved configuration and completion evidence."""
import json
from experiments.cvs_feature_disentangle import design as d
from experiments.cvs_phase1_stack.publish import ssh
ART=d.ROOT/'local_artifacts'/d.RELEASE

def inspect():
    script='''
import json,time
from pathlib import Path
b=Path(BASE);release=Path(RELEASE)
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def proc(pid):
 p=Path('/proc')/str(pid)
 try:return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=p.joinpath('cmdline').read_bytes().decode().split('\\0'),state=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[0])
 except OSError:return None
receipt=read(b/'dispatcher_active.json') or read(release/'submit.json');rows=[]
for r in (read(b/'launch.json') or dict(rows=[]))['rows']:
 log=Path(r['log']);text=log.read_text(errors='replace') if log.is_file() else ''
 root=b/r['row_id'];args=read(root/'source/resolved_native_args.json')
 rows.append(dict(**r,process=proc(r['pid']),log_bytes=len(text.encode()),tail=text.splitlines()[-3:],
 resolved=read(root/'source/resolved_config.json'),actual_recipe={k:args.get(k) for k in ['use_unlabeled','use_ema_teacher','pseudo_temporal_mode','phase1_lr_schedule','sat_training_mode','use_concat_sat_channel_aug','lambda_sat_cls','lambda_sat_cons','label_epochs','pseudo_epochs','label_smoothing']} if args else None,
 source_completion=read(root/'source/completion.json'),prediction_completion=read(root/'prediction/complete.json'),completion=read(root/'completion.json'),failure=read(root/'failure.json')))
print(json.dumps(dict(at=time.time(),receipt=receipt,controller=proc(receipt['pid']) if receipt else None,
 queue=read(b/'queue_state.json'),selection=read(b/'source_selection.json'),failure=read(b/'failure.json'),completion=read(b/'completion.json'),rows=rows,
 preflight=read(release/'preflight-synthetic/completion.json'))))
'''.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE))
    s=json.loads(ssh(script));d.write(ART/'readback.json',s)
    print(json.dumps({k:v for k,v in s.items() if k!='rows'}))
    for r in s['rows']:print(json.dumps({k:r[k] for k in ['row_id','pid','gpu','process','log_bytes','tail','actual_recipe','completion','failure']}))
    return s

if __name__=='__main__':inspect()
