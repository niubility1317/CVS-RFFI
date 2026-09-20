import subprocess,json
from pathlib import Path
root=Path('E:/type10-7');wt=root/'code/snapshots/daot_practical_three_20260918_wt';run='20260918-phase1-daot-rc4-practical4-manysig-s392005-r03';r=root/'automation_reports/CV-SincNet'/run
spec=json.loads((root/'automation_reports/CV-SincNet/20260919-phase1-daot-rc4-practical4-test-s392005-r02/experiment.json').read_text(encoding='utf-8'))
for row in spec['rows']:
 row['epochs']={'FULL_NOEQ':130,'FULL_ZF':110,'RESIDUAL_NOEQ':200,'FULL_MMSE':140}[row['row_id'].removeprefix('DAOT_RC4_PRACTICAL_').removesuffix('_s392005')]
source=(wt/'experiments/adv3b02_xuc/tools/recount_practical_snapshot.py').read_text(encoding='utf-8').replace("if __name__ == '__main__':",'if False:')
script=source+'\nspec='+repr(spec)+"\nprint(json.dumps({'timestamp':datetime.datetime.now().astimezone().isoformat(),'status':'VERIFIED','rows':{r['row_id']:recount(r) for r in spec['rows']}}))\n"
x=subprocess.run(['ssh','-F',str(root/'tools/n607_ssh_config'),'-o','BatchMode=yes','-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
d=json.loads(x.stdout);(r/'evidence/latest_test_recount_20260920.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
for k,v in d['rows'].items():print(k,v['epoch'],{s:{m:round(a[m]*100,4) for m in ('accuracy','macro_f1')} for s,a in v['metrics'].items()})
script='''import torch,json,sys
from pathlib import Path
sys.path.insert(0,'/home/szu2070436088/2510044040/CV-SincNet/releases/daot_rc4_practical4_20260918_f1ea4a5fb6/code')
p=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20260918-phase1-daot-rc4-practical4-manysig-s392005-r03/DAOT_RC4_PRACTICAL_RESIDUAL_NOEQ_s392005')
a=torch.load(p/'epoch_200_ssdg.pth',map_location='cpu');b=torch.load(p/'final_ssdg.pth',map_location='cpu')
print('KEYS',a.keys(),b.keys())
for k in a:
 if isinstance(a[k],dict) and any(torch.is_tensor(v) for v in a[k].values()):
  print('TENSOR_MAPPING',k,'equal',set(a[k])==set(b[k]) and all(torch.equal(v,b[k][n]) for n,v in a[k].items() if torch.is_tensor(v)))
print('EPOCHS',a.get('epoch'),b.get('epoch'))
print('COMPLETION', (p/'completion.json').read_text())
'''
x=subprocess.run(['ssh','-F',str(root/'tools/n607_ssh_config'),'-o','BatchMode=yes','-T','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=script.encode(),capture_output=True,check=True)
(r/'evidence/residual_final_equivalence_20260920.txt').write_bytes(x.stdout);print(x.stdout.decode())
