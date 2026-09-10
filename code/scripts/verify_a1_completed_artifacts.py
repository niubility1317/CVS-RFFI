"""Read checkpoint metadata and artifact completion for newly finished runs."""
import argparse,json,subprocess
from pathlib import Path

REMOTE=r'''
import json,gc,datetime,sys
from pathlib import Path
import torch
p=Path('/home/szu2070436088/2510044040/CV-SincNet')
result={}
for run,row,expected in [
    ('a1_r3_budget_s392005_20260909_r2','R3_B160',160),
    ('a1_r3_clean_cross_rx_s392005_20260909_r1','R3_REFERENCE_CLEAN_CROSS_RX',200)]:
    folder=p/'runs'/run/row
    release=Path(json.loads((p/'runs'/run/'pipeline_state.json').read_text())['release'])
    sys.path.insert(0,str(release/'code'))
    checkpoint=folder/'final_ssdg.pth'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    args=saved.get('args',{})
    result[row]={'checkpoint':str(checkpoint),'bytes':checkpoint.stat().st_size,
        'modified':datetime.datetime.fromtimestamp(checkpoint.stat().st_mtime,datetime.timezone.utc).isoformat(),
        'epoch':saved.get('epoch'),'expected_epoch':expected,'model_tensors':len(saved.get('model',{})),
        'args':{k:args.get(k) for k in ['from_scratch','a1_scratch_only','baseline_ckpt','teacher_ckpt',
            'epochs','seed','wisig_train_rxs','wisig_train_days','wisig_test_rxs','split_mode',
            'labeled_ratio','unlabeled_ratio','source_val_ratio','checkpoint_selection','best_metric']},
        'target_score_count':len(list((folder/'target_epochs').glob('E*/score.json'))) if (folder/'target_epochs').exists() else 0}
    assert saved.get('epoch')==expected and args.get('from_scratch') and not args.get('baseline_ckpt') and not args.get('teacher_ckpt')
    del saved;gc.collect()
folder=p/'runs/tweak_config2_portability_20260909_v5/official_config2_full'
results=json.loads((folder/'results.json').read_text())
result['LoRa']={'status':results['status'],'checkpoint_exists':Path(results['checkpoint']).is_file(),
    'completed_epochs':len(results['training']['epoch_rows']),
    'single_domain_rows':len(results['single_domain_calibration_4x4']),
    'multiple_domain_rows':len(results['multiple_configuration_calibration'])}
result['verified_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
print(json.dumps(result))
'''

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','N607',
        '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -'],input=REMOTE,text=True,
        capture_output=True,encoding='utf-8',timeout=60)
    if response.returncode: raise RuntimeError(response.stderr)
    value=json.loads(response.stdout);args.output.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value))
