import json,subprocess
from pathlib import Path
root=Path('E:/type10-7'); run='20260918-phase1-daot-rc4-original-leo-manysig-s392005-r01'
r=root/'automation_reports/CV-SincNet'/run
c=json.loads((r/'evidence/launch.json').read_text(encoding='utf-8'))
script='''import json,datetime
from pathlib import Path
p=Path(OUTPUT)/'target_epochs/E200'
score=json.loads((p/'score.json').read_text())
pred=json.loads((p/'predictions.json').read_text())
truth=json.loads(Path(score['truth_path']).read_text())
labels={r['sample_id']:r['label'] for r in truth['records']}
assert len(labels)==len(truth['records'])==168000
scenes=['clean','clear_leo','low_elev_leo','rain_leo']
cm={s:[[0]*6 for _ in range(6)] for s in scenes}; seen={s:set() for s in scenes}
for row in pred['records']:
 s=row['scenario']; sid=row['sample_id']; y=labels[sid]; yh=row['predicted_class']
 assert sid not in seen[s] and 0<=y<6 and 0<=yh<6
 seen[s].add(sid);cm[s][y][yh]+=1
metrics={}
for s in scenes:
 assert seen[s]==set(labels)
 mat=cm[s]; total=sum(map(sum,mat)); correct=sum(mat[i][i] for i in range(6))
 f1=[2*mat[i][i]/(sum(mat[i])+sum(row[i] for row in mat)) for i in range(6)]
 assert correct==score['metrics'][s]['correct'] and total==score['metrics'][s]['total']
 metrics[s]={'total':total,'correct':correct,'accuracy':correct/total,'macro_f1':sum(f1)/6,'per_class_f1':f1,'confusion_matrix_true_rows':mat}
assert len(pred['records'])==672000
print(json.dumps({'status':'VERIFIED','timestamp':datetime.datetime.now().astimezone().isoformat(),'checkpoint':json.loads((p/'evaluation_scope.json').read_text())['checkpoint'],'prediction_path':str(p/'predictions.json'),'truth_path':score['truth_path'],'record_count':len(pred['records']),'unique_physical_samples':len(labels),'coverage_exact':True,'existing_score_matches':True,'scope':'fixed E200 frozen prediction recount; no new inference; no model selection; not blind confirmation','metrics':metrics},indent=2))
'''.replace('OUTPUT',repr(c['output']))
res=subprocess.run(['ssh','-F',str(root/'tools/n607_ssh_config'),'-o','BatchMode=yes','-T','N607','python3 -'],input=script.encode(),capture_output=True,check=True)
data=json.loads(res.stdout)
(r/'evidence/e200_clean_original_leo_recount_20260919.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(data,ensure_ascii=False,indent=2))
