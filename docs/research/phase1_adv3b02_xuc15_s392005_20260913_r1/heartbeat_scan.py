"""Read-only operational scan. No target truth or model states are loaded."""
import argparse,json,math,re,time
from pathlib import Path
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
run=project/'runs/phase1_adv3b02_xuc15_s392005_20260913_r1'
state=json.loads((run/'pipeline_state.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--rows',default='')
selected=set(parser.parse_args().rows.split(','))-{''}
result={'time':time.time(),'status':state['status'],'rows':{},'errors':[]}

def counters(obj,result):
    if isinstance(obj,dict):
        for key,value in obj.items():
            if key in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad') and isinstance(value,(int,float)):
                result[key]=max(result.get(key,0),value)
            counters(value,result)
    elif isinstance(obj,list):
        for value in obj:counters(value,result)

for rid,entry in state['rows'].items():
    if selected and rid not in selected:continue
    row={'status':entry['status'],'lines':{},'numerical_errors':0,'unaccepted_steps':0,'actions':{},'native_nonfinite_max':{}}
    folder=run/rid
    for name in ('actions.jsonl','logs.jsonl','metrics_epoch.jsonl'):
        path=folder/name
        if not path.is_file():continue
        count=0
        with path.open() as stream:
            for raw in stream:
                if not raw.endswith('\n'):continue
                item=json.loads(raw);count+=1
                for key in ('loss','mean_loss','grad_norm'):
                    value=item.get(key)
                    if isinstance(value,(int,float)) and not math.isfinite(value):row['numerical_errors']+=1
                if name=='actions.jsonl':
                    row['unaccepted_steps']+=int(item.get('accepted') is not True)
                    action=item.get('action','UNKNOWN');row['actions'][action]=row['actions'].get(action,0)+1
                if name=='metrics_epoch.jsonl':
                    counters(item,row['native_nonfinite_max'])
                    row['native_epoch']=item.get('epoch')
        row['lines'][name]=count
    log=project/'logs'/run.name/(rid+'.train.log')
    text=log.read_text(errors='replace') if log.exists() else ''
    fatal=[marker for marker in ('Traceback (most recent call last)','CUDA out of memory','RC4-NONFINITE-GUARD-STOP') if marker in text]
    epochs=re.findall(r'\[EPOCH-END\] E(\d+)/200',text)
    row['completed_epoch']=int(epochs[-1]) if epochs else None
    row['fatal_markers']=fatal
    row['latest_checkpoint_present']=(folder/'latest_ssdg.pth').is_file()
    row['final_checkpoint_present']=(folder/'final_ssdg.pth').is_file()
    if row['numerical_errors'] or row['unaccepted_steps'] or fatal or entry['status'] not in ('RUNNING','QUEUED','TRAINING_COMPLETE','PREDICTIONS_FIXED','SCORED'):
        result['errors'].append(rid)
    result['rows'][rid]=row
print(json.dumps(result,indent=2))
