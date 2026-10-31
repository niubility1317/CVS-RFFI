import json
from pathlib import Path
import pytest
from experiments.pamf_phase2_practical import milkr_r42_main as main


def config(tmp_path):
    rx=['1-1','14-7','2-1','20-1','7-14','7-7','8-8']
    return dict(kernel_method='adaptive_residual_ensemble',arm='R42',K=[5],N=[20],
        model_seeds=[2026092701,2026092702,2026092703,2026092704],target_receivers=rx,
        capsules=[f'capsules/{p}/MIX{i:02d}/{r}' for p in ('P01','P02','P03') for i in range(1,9) for r in rx],
        run_root=str(tmp_path/'run'),data_root=str(tmp_path/'data'),pilot_run_root=str(tmp_path/'pilot'))


def test_complete_main_matrix_and_isolated_task_roots(tmp_path):
    rt=config(tmp_path);matrix=main.tasks(rt)
    assert len(matrix)==672 and len({t['key'] for t in matrix})==672
    roots=[main.task_runtime(rt,t) for t in matrix]
    assert len({r['run_root'] for r in roots})==672
    assert all(r['model_seeds']==[t['seed']] and r['capsules']==[t['capsule']] for r,t in zip(roots,matrix))
    for wrong in (dict(rt,K=[1,5]),dict(rt,capsules=rt['capsules'][:-1]),dict(rt,model_seeds=[2026092701]),dict(rt,arm='R43')):
        with pytest.raises(Exception):main.tasks(wrong)


def test_main_cannot_score_before_prediction_closure(tmp_path,monkeypatch):
    rt=config(tmp_path);calls=[]
    monkeypatch.setattr(main,'queue',lambda rt,items,action,per_gpu:calls.append(action))
    monkeypatch.setattr(main,'prediction_closed',lambda *args:(_ for _ in ()).throw(RuntimeError('missing prediction')))
    monkeypatch.setattr(main.subprocess,'run',lambda *a,**k:calls.append('score'))
    with pytest.raises(RuntimeError,match='missing prediction'):main.dispatch(rt)
    assert calls==['source','worker']
    assert not (Path(rt['run_root'])/'predictions_complete.json').exists()


def pilot_files(rt):
    task=main.tasks(rt)[0];old=Path(rt['pilot_run_root'])/'predictions/1-1';old.mkdir(parents=True)
    cap=Path(rt['data_root'])/task['capsule'];(cap/'splits').mkdir(parents=True)
    main.write(cap/'manifest.json',dict(capsule_id='capsule1'))
    rows=[]
    for n,stage in ((0,'B'),(20,'C')):
        spec=dict(k=5,new_count=n,split_id=f'split{n}',registered_classes=['old'] if not n else ['old','new'])
        main.write(cap/'splits'/f'split{n}.json',spec)
        for s in (0,1):
            for q in (0,1,2):
                name=f'{stage}-K5-N{n}-S{s}Q{q}-R42.npz';(old/name).write_bytes(b'fixed prediction')
                rows.append(dict(**spec,stage=stage,input_path=f'S{s}Q{q}',arm='R42',prediction=name))
    main.write(old/'predictions.json',dict(capsule=task['capsule'],capsule_id='capsule1',model_seed=task['seed'],rows=rows))
    main.write(old/'complete.json',dict(status='PREDICTIONS_COMPLETE',rows=120,truth_read=False,query_fit=False))
    return task,old,cap


def test_reuse_fixed_pilot_main_rows_without_query_access(tmp_path):
    rt=config(tmp_path);task,old,cap=pilot_files(rt);jobrt=main.task_runtime(rt,task)
    assert main.reuse_pilot(rt,task,jobrt)
    main.prediction_closed(rt,task)
    output=Path(jobrt['run_root'])/'predictions/1-1'
    assert len(list(output.glob('*.npz')))==12
    assert main.read(output/'complete.json')['reused_existing_predictions']
    assert not main.reuse_pilot(rt,dict(task,seed=2026092702),jobrt)


def test_pilot_wrong_split_prevents_reuse(tmp_path):
    rt=config(tmp_path);task,old,cap=pilot_files(rt)
    spec=main.read(cap/'splits/split20.json');spec['split_id']='other';main.write(cap/'splits/split20.json',spec,exclusive=False)
    with pytest.raises(Exception,match='Pilot split differs'):main.reuse_pilot(rt,task,main.task_runtime(rt,task))
    assert not (Path(main.task_runtime(rt,task)['run_root'])/'predictions').exists()


def test_new_source_uses_matching_F_seed_and_rejects_target_ancestry(tmp_path,monkeypatch):
    from experiments.pamf_phase2_practical import features,milkr_orbit_source
    rt=config(tmp_path);seen=[]
    monkeypatch.setattr(features,'source_smoke',lambda local,out:seen.append((local['model_seeds'],local['kernel_method'])))
    def calibrate(local):
        audit=dict(steps=600,target_reads=1,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0)
        main.write(Path(local['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',target_eligible=True,
            source_metric_training=[dict(C4=audit,C8=audit) for _ in range(6)],K1_rule={}))
    monkeypatch.setattr(milkr_orbit_source,'calibrate',calibrate)
    with pytest.raises(Exception,match='Illegal source training input'):main.source(rt,2026092702)
    assert seen==[([2026092702],'crossrx_residual_ensemble')]
    assert not (main.source_root(rt,2026092702)/'main_source_complete.json').exists()


def test_queue_updates_progress_and_limits_two_workers_per_card(tmp_path,monkeypatch):
    rt=config(tmp_path);rt.update(allowed_gpus=[0,1],minimum_free_MiB=6000,poll_seconds=0,runtime_file='runtime.json')
    root=Path(rt['run_root']);(root/'logs').mkdir(parents=True);(root/'launches').mkdir()
    live={0:0,1:0};peak={0:0,1:0};closed=[]
    class Process:
        def __init__(self,cmd,env,**kwargs):
            self.gpu=int(env['CUDA_VISIBLE_DEVICES']);self.pid=100+len(closed);self.polls=0;self.ended=False
            live[self.gpu]+=1;peak[self.gpu]=max(peak[self.gpu],live[self.gpu])
        def poll(self):
            self.polls+=1
            if self.polls<4:return None
            if not self.ended:live[self.gpu]-=1;self.ended=True
            return 0
    monkeypatch.setattr(main.subprocess,'Popen',Process)
    monkeypatch.setattr(main.subprocess,'check_output',lambda *a,**k:'24000')
    monkeypatch.setattr(main,'prediction_closed',lambda rt,t:closed.append(t['key']))
    main.queue(rt,[dict(key=f'j{i}') for i in range(9)],'worker',2)
    assert peak=={0:2,1:2} and live=={0:0,1:0} and len(closed)==9
    assert main.read(root/'worker_state.json')['status']=='COMPLETE'
