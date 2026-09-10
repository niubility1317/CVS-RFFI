import json,subprocess
from pathlib import Path
REMOTE=r'''
import json,datetime
from pathlib import Path
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
result={'captured_at':datetime.datetime.now().isoformat(),'runs':{},'pending_scores':[]}
for name in ('a1_extended_s392005_20260910_r1','a1_mechanism_periodic_s392005_20260910_r1'):
 root=p/'runs'/name;state=json.loads((root/'pipeline_state.json').read_text());scores=[]
 for score in sorted(root.glob('*/target_epochs/E*/score.json')):
  scope=score.with_name('evaluation_scope.json')
  if not scope.exists():result['pending_scores'].append(str(score));continue
  scores.append({'row':score.parent.parent.parent.name,'epoch':int(score.parent.name[1:]),
   'score_path':str(score),'score':json.loads(score.read_text()),'scope':json.loads(scope.read_text()),
   'predictions_exist':score.with_name('predictions.json').is_file()})
 result['runs'][name]={'state':state,'scores':scores}
lora=p/'runs/tweak_config2_portability_20260910_v7/official_config2_full/results.json'
result['lora_v7']=json.loads(lora.read_text()) if lora.exists() else None
print(json.dumps(result))
'''
d=json.loads(subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3 -'],
    input=REMOTE,text=True,encoding='utf-8',timeout=30))
out=Path(__file__).with_name('a1_latest_tests_20260910_1623.json');out.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
print(d['captured_at'])
print({name:len(run['scores']) for name,run in d['runs'].items()})
print('lora keys',list(d['lora_v7']))
print('score sample',next(iter(d['runs'].values()))['scores'][:1])
