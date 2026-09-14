import ast,json,gzip,subprocess
from pathlib import Path
base=Path('E:/type10-7')
src=base/'github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913/local_validation/audit_full_results.py'
tree=ast.parse(src.read_text(encoding='utf-8'))
code=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE' for t in n.targets))
compile(code,'reviewed_readonly_full_audit','exec')
out=base/'automation_reports/CV-SincNet/daot_fasttrust_game_comprehensive_20260914'
p=subprocess.run(['ssh','-F',str(base/'tools/n607_ssh_config'),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],input=code.encode('utf-8'),capture_output=True,timeout=180)
if p.returncode:
    raise RuntimeError(p.stderr.decode('utf-8','replace'))
d=json.loads(p.stdout)
(out/'full_live_audit.json.gz').write_bytes(gzip.compress(p.stdout))
print(json.dumps({'read_at':d['read_at'],'state':d['full']['state']['status'],'rows':{k:{'status':v['state']['status'],'epochs':v.get('logs.jsonl',{}).get('count'),'counts':v.get('action_audit',{}).get('counts'),'actions':v.get('action_audit',{}).get('actions'),'score':bool(v['score'])} for k,v in d['full']['rows'].items()}},ensure_ascii=False))
