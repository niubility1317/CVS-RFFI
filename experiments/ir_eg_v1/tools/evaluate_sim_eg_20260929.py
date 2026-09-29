"""Fixed E200 SIM/EG evaluation: complete both predictions before truth scoring."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text())
    assert [r['row_id'] for r in spec['rows']]==['SIM_s392005','EG_s392005']
    run=Path(spec['execution']['remote_run_root']);logs=Path(spec['execution']['remote_log_root'])
    if run.exists() or logs.exists():raise FileExistsError('Evaluation exists; reconcile before retry')
    release=Path(spec['execution']['training_release'])
    sys.path.insert(0,str(release/'code'))
    import torch
    from cvsrffi.xuc_fusion.checkpoints import load_model
    torch.set_num_threads(2)
    uuids=dict(line.split(', ') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).strip().splitlines())
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    free=dict(line.split(', ') for line in subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines())
    for row in spec['rows']:
        # One existing job plus one authorized inference fits the two-job cap.
        uuid=uuids[str(row['gpu'])]
        assert sum(line.startswith(uuid+',') for line in apps.splitlines())<2,'requested GPU has two jobs'
        assert int(free[str(row['gpu'])])>=8192,'insufficient free memory for inference'
    run.mkdir(parents=True,exist_ok=False);logs.mkdir(parents=True,exist_ok=False)
    state=dict(status='CHECKPOINT_VALIDATION',commit=a.commit,source_run=spec['parent_run_ids'][0],pid=os.getpid(),rows={},started=time.time(),target_feedback=False)
    def save():
        temp=run/'state.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(run/'evaluation_state.json')
    save()
    try:
        contract=json.loads(Path(spec['execution']['source_contract']).read_text())
        for row in spec['rows']:
            folder=Path(row['checkpoint']).parent
            completion=json.loads((folder/'completion.json').read_text())
            assert completion['epochs']==200 and completion['steps']==44400 and completion['status']=='TRAINING_COMPLETE'
            model,payload=load_model(row['checkpoint'],torch.device('cpu'))
            assert payload['row']['id']==row['row_id'] and payload['step']==44400
            assert payload['source_info']==contract and payload['target_contact'] is False
            assert payload['initialization']=='scratch_only' and payload.get('resume_lineage') is None
            with torch.no_grad():assert torch.isfinite(model(torch.zeros(2,2,256),return_aux=True)['tx_logits']).all()
            state['rows'][row['row_id']]=dict(status='FROZEN_E200_VERIFIED',checkpoint=row['checkpoint'],source_contract_equal=True,checkpoint_smoke='PASS')
            del model,payload
        save()
        inputs=Path(spec['execution']['input_package'])
        manifest=json.loads((inputs/'manifest.json').read_text())
        assert manifest['record_count']==168000 and manifest['contains_labels'] is False
        ids=set(manifest['sample_ids']);assert len(ids)==168000
        active=[]
        for row in spec['rows']:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(row['gpu']),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
            handle=Path(row['log_path']).open('x')
            child=subprocess.Popen(row['command'],cwd=release,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT)
            active.append((row,child,handle))
            state['rows'][row['row_id']].update(status='PREDICTING',pid=child.pid,gpu=row['gpu'],command=row['command'])
        state['status']='PREDICTING';save()
        print('SIM and EG predictions started; batch=256 seed=392005 scenes=clean+three_LEO target_truth=unread',flush=True)
        for row,child,handle in active:
            code=child.wait();handle.close()
            if code:raise RuntimeError(f"Prediction failed {row['row_id']} exit={code}")
            pred=Path(row['output_root'])/'predictions.json';doc=json.loads(pred.read_text())
            assert doc['record_count']==672000 and len(doc['records'])==672000
            for scene in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:
                actual=[r['sample_id'] for r in doc['records'] if r['scenario']==scene]
                assert len(actual)==168000 and set(actual)==ids
            assert all(r['run_id']==spec['run_id'] and r['row_id']==row['row_id'] for r in doc['records'])
            del doc
            state['rows'][row['row_id']].update(status='PREDICTIONS_FIXED',record_count=672000,prediction=str(pred))
            save();print(row['row_id']+' PREDICTIONS_FIXED',flush=True)
        state['status']='ALL_PREDICTIONS_FIXED';state['prediction_freeze_time']=time.time();save()
        # This is the first operation authorized to connect truth. Each scorer
        # is a separate process and its output cannot change either checkpoint.
        for row in spec['rows']:
            state['status']='SCORING';save()
            with (logs/(row['row_id']+'.score.log')).open('x') as handle:
                subprocess.run(row['score_command'],cwd=release,stdout=handle,stderr=subprocess.STDOUT,check=True,
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'))
            score=json.loads(Path(row['score_output']).read_text())
            assert score['overall']['n']==672000
            state['rows'][row['row_id']].update(status='SCORED',score=row['score_output'])
            save();print(row['row_id']+' SCORED',flush=True)
        state.update(status='SCORED_COMPLETE',finished=time.time());save()
    except Exception as exc:
        state.update(status='FAILED',error=repr(exc));save();raise


if __name__=='__main__':main()
