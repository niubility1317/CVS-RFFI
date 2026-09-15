"""Evaluate the frozen completed inventory locally; N607 access is read-only."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT='/home/szu2070436088/2510044040/CV-SincNet'
TARGET='phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4'
CONNECTION=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--reuse-base',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    inventory=json.loads(a.inventory.read_text(encoding='utf-8'));run=inventory['state']['run_id']
    rows=[k for k,v in inventory['state']['rows'].items() if v['status']=='TRAINING_COMPLETE']
    if not rows:raise ValueError('no completed rows')
    reused=set()
    if a.reuse_base:
        prior=json.loads((a.reuse_base/'state.json').read_text())
        assert prior['status']=='SCORED_COMPLETE' and prior['run_id']==run
        assert prior['batch_size']==256 and prior['seed']==392005
        reused=set(prior['rows'])&set(rows)
    state=dict(status='COPYING',run_id=run,rows=rows,created=time.time(),target_feedback=False,
        scope='previously_exposed_benchmark_descriptive_evaluation',device='local RTX5070Ti',batch_size=256,seed=392005,results={},reused_rows=sorted(reused))
    def save():
        (a.output/'state.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    def copy(remote,local):
        if local.exists():raise FileExistsError(local)
        subprocess.run(['scp',*CONNECTION,'N607:'+remote,str(local)],check=True)
    save();inputs=a.output/'target_inputs';inputs.mkdir()
    for name in ('iq.npy','manifest.json'):
        if a.reuse_base:shutil.copy2(a.reuse_base/'target_inputs'/name,inputs/name)
        else:copy(f'{PROJECT}/runs/{TARGET}/target_inputs/{name}',inputs/name)
    manifest=json.loads((inputs/'manifest.json').read_text());assert manifest['record_count']==168000 and manifest['contains_labels'] is False
    contract=a.output/'source_contract.json';copy(f'{PROJECT}/runs/{run}/source_contract.json',contract)
    expected=json.loads(contract.read_text())
    for rid in rows:
        folder=a.output/rid
        if rid in reused:
            shutil.copytree(a.reuse_base/rid,folder)
            pred=json.loads((folder/'prediction/predictions.json').read_text());assert pred['record_count']==672000
            assert all(r['run_id']==run and r['row_id']==rid for r in pred['records'])
            state['results'][rid]=dict(status='SCORED',count=672000,reused_from=str(a.reuse_base/rid))
        else:
            folder.mkdir()
            for name in ('final_ssdg.pth','completion.json','initialization.json','resolved_config.json','resolved_dr_config.json','source_contract.json','logs.jsonl'):
                copy(f'{PROJECT}/runs/{run}/{rid}/{name}',folder/name)
        c=json.loads((folder/'completion.json').read_text());assert c['status']=='TRAINING_COMPLETE' and c['epochs']==200 and c['steps']==44400
        cfg=json.loads((folder/'resolved_config.json').read_text());assert cfg['from_scratch'] and not cfg['baseline_ckpt'] and not cfg['game_resume']
        assert json.loads((folder/'source_contract.json').read_text())['role_ids']==expected['role_ids']
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONIOENCODING='utf-8')
    # Check all final checkpoint provenance and rebuild without accessing query.
    sys.path.insert(0,str(ROOT/'code'))
    import torch
    from cvsrffi.xuc_fusion.checkpoints import load_model
    torch.set_num_threads(2)
    state['torch']=torch.__version__
    for rid in rows:
        model,payload=load_model(a.output/rid/'final_ssdg.pth',torch.device('cpu'))
        assert payload['row']['id']==rid
        assert not payload.get('synthetic_diagnostic_only',False) and payload['step']==44400
        assert payload['source_info']['role_ids']==expected['role_ids']
        with torch.no_grad():assert torch.isfinite(model(torch.zeros(2,2,256),return_aux=True)['tx_logits']).all()
        del model,payload
    state['status']='PREDICTING';state['checkpoint_provenance']='VERIFIED_SCRATCH_MATCHED_SOURCE_ROLES_FIXED_E200';save()
    for rid in rows:
        if rid in reused:continue
        folder=a.output/rid;cmd=[sys.executable,'-X','utf8',str(ROOT/'code/scripts/predict_phase1_truth_last.py'),
            '--checkpoint',str(folder/'final_ssdg.pth'),'--output-root',str(folder/'prediction'),'--input-package',str(inputs),
            '--run-id',run,'--row-id',rid,'--recipe',str(ROOT/'configs/core90_recipe_reference.json'),
            '--mode','predict','--batch-size','256','--num-workers','0','--device','cuda:0']
        with (folder/'predict.log').open('x',encoding='utf-8') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        pred=json.loads((folder/'prediction/predictions.json').read_text());assert pred['record_count']==672000
        state['results'][rid]=dict(status='PREDICTIONS_FIXED',count=672000);save();print(rid+' PREDICTIONS_FIXED',flush=True)
    state['status']='ALL_PREDICTIONS_FIXED';save()
    truth=a.output/'truth_sidecar.json';copy(f'{PROJECT}/runs/{TARGET}/target_truth/truth_sidecar.json',truth)
    state['status']='SCORING';save()
    for rid in rows:
        if rid in reused:continue
        folder=a.output/rid
        cmd=[sys.executable,'-X','utf8',str(ROOT/'tools/score_native_joint_predictions.py'),'--predictions',str(folder/'prediction/predictions.json'),
            '--truth',str(truth),'--output',str(folder/'score.json')]
        with (folder/'score.log').open('x',encoding='utf-8') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        state['results'][rid]['status']='SCORED';save();print(rid+' SCORED',flush=True)
    state.update(status='SCORED_COMPLETE',finished=time.time());save()

if __name__=='__main__':main()
