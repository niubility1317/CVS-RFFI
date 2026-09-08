"""Read-only N607 startup evidence; never loads model weights into a model."""
import argparse
import json
from pathlib import Path
import subprocess

REMOTE = r'''
import csv,json,math,os,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
import torch
project=Path('/home/szu2070436088/2510044040/CV-SincNet')
run_id=sys.argv[1]
root=project/'runs'/run_id
logs=project/'logs'/run_id
def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else None
state=read_json(root/'pipeline_state.json')
sys.path.insert(0,str(Path(state['release'])/'code'))
result={'captured_utc':datetime.now(timezone.utc).isoformat(),'pipeline':state,
        'matrix':read_json(root/'effective_matrix.json'),'rows':{},
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True),
        'compute':subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True),
        'cuda_execution':read_json(logs/'scratch_execution.json')}
for name,entry in state['rows'].items():
    folder=root/name; pid=int(entry['pid']); proc=Path('/proc')/str(pid)
    log=logs/(name+'.train.log')
    lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    row={'pid_alive':proc.exists(),'cwd':os.readlink(proc/'cwd') if proc.exists() else None,
         'process':read_json(folder/'process.json'),'launch_config':read_json(folder/'launch_config.json'),
         'initialization':[line for line in lines if line.startswith('[SSDG-TRAIN]')],
         'log_tail':lines[-8:],'metrics_rows':0}
    metrics=folder/'metrics_epoch.csv'
    if metrics.is_file():
        records=list(csv.DictReader(metrics.open()))
        row['metrics_rows']=len(records)
        if records:
            row['latest_metrics']={k:v for k,v in records[-1].items() if
                'r3_' in k or 'nonfinite' in k or 'optimizer_step' in k or k in ('epoch','train_loss','epoch_time_s')}
            row['latest_nan_keys']=[k for k,v in records[-1].items() if str(v).lower() in ('nan','inf','-inf')]
    anomaly=folder/'first_rc4_anomaly.pt'
    if anomaly.is_file():
        # Diagnostic packet only; inspect metadata without constructing/loading a model.
        packet=torch.load(anomaly,map_location='cpu')
        row['first_anomaly']={k:packet.get(k) for k in ('epoch','batch_index','loss_finite','skipped_nonfinite_loss',
            'skipped_nonfinite_grad','loss','first_nonfinite_gradient','scaler')}
        del packet
    result['rows'][name]=row
print(json.dumps(result,default=str))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='a1_r3_scratch_s392005_20260908_r1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not args.run_id.replace('_','').isalnum(): raise ValueError('Invalid run ID')
    result=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607',
        '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-',args.run_id],
        input=REMOTE,text=True,encoding='utf-8',capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    payload=json.loads(result.stdout)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'pipeline':payload['pipeline'],'rows':{k:{
        'alive':v['pid_alive'],'metrics_rows':v['metrics_rows'],
        'latest_metrics':{key:value for key,value in v.get('latest_metrics',{}).items() if key in
            ('epoch','epoch_time_s','train_loss','train_optimizer_step_applied','train_skipped_nonfinite_grad',
             'train_skipped_nonfinite_loss','train_r3_l_s_loss','train_r3_u_s_loss')},
        'first_anomaly_batch':v.get('first_anomaly',{}).get('batch_index')}
        for k,v in payload['rows'].items()}},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
