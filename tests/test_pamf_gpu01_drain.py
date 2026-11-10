from experiments.pamf_split_spectrum_paper import gpu01_drain


def test_no_new_work_on_gpu_zero_or_one():
    assert tuple(gpu01_drain.dispatch_gpus()) == (2,3,4,5,6,7)
    assert [gpu01_drain.limits(i) for i in range(8)] == [0,0,3,3,3,3,3,1]


def test_only_current_exact_controller_may_be_handed_off():
    rt={'source_control_release':'/control','source_control_module':'experiments.pamf_split_spectrum_paper.speed_pilot','python':'/env/python'}
    argv=['/env/python','-u','-m','experiments.pamf_split_spectrum_paper.speed_pilot',
          '--config','/control/runtime.json']
    assert gpu01_drain.matches_dispatcher(rt,'/control',argv)
    assert not gpu01_drain.matches_dispatcher(rt,'/different',argv)
    assert not gpu01_drain.matches_dispatcher(rt,'/control',argv+['other'])


def test_interrupted_training_gets_new_output_and_fixed_dependencies(tmp_path):
    from pathlib import Path
    old=tmp_path/'F/row/source'
    job={'id':'train__row','kind':'train','config':'old.json','completion':str(old/'complete.json')}
    dep={'id':'predict__row','kind':'predict','config':'pred.json','dependencies':[str(old/'complete.json')]}
    source={'source_output':str(old),'predictions':'pred','scores':'score'}
    cfg={'row_id':'row','model_seed':42,'output_root':str(old)}
    pcfg={'row_id':'row','source_output':str(old)}
    new, new_dep, new_source, new_cfg, new_pcfg = gpu01_drain.replacement_plan(job,dep,source,cfg,pcfg,tmp_path)
    assert new['id']=='train__row--gpu01-r01' and new_cfg['model_seed']==42
    assert new_cfg['output_root']!=str(old) and Path(new_cfg['output_root']).name=='source-gpu01-r01'
    assert new_source['source_output']==new_pcfg['source_output']==new_cfg['output_root']
    assert new_dep['dependencies']==[new['completion']]
    assert cfg['output_root']==str(old) and dep['dependencies']==[str(old/'complete.json')]


def test_interrupted_prediction_repoints_independent_scoring(tmp_path):
    old=tmp_path/'readouts/row/predictions'
    job={'id':'predict__row','kind':'predict','config':'pred.json','completion':str(old/'complete.json')}
    dep={'id':'score__row','kind':'score','config':'score.json','dependencies':[str(old/'complete.json')]}
    source={'source_output':'source','predictions':str(old),'scores':'score'}
    cfg={'row_id':'row','output_root':str(old)}
    scfg={'row_id':'row','prediction_output':str(old)}
    new,new_dep,new_source,new_cfg,new_scfg=gpu01_drain.replacement_plan(job,dep,source,cfg,scfg,tmp_path)
    assert new_source['predictions']==new_scfg['prediction_output']==new_cfg['output_root']
    assert new_dep['dependencies']==[new['completion']]


def test_successor_handoff_record_is_exclusive_and_validated(tmp_path,monkeypatch):
    import json
    from experiments.pamf_split_spectrum_paper import link_gpu01_handoff as link
    previous=tmp_path/'old';current=tmp_path/'new';previous.mkdir();current.mkdir()
    (previous/'pipeline_state.json').write_text(json.dumps({'pid':10}),encoding='utf8')
    state={'status':'RUNNING','pid':11,'previous_pid':10,'dispatch_gpus':[2,3,4,5,6,7],'jobs':{}}
    (current/'pipeline_state.json').write_text(json.dumps(state),encoding='utf8')
    monkeypatch.setattr(link,'alive',lambda pid:pid==11)
    result=link.link(previous,current)
    assert result['control_root']==str(current)
    assert link.link(previous,current)==result


def test_takeover_persists_retry_without_losing_existing_jobs(tmp_path, monkeypatch):
    import json
    import pathlib
    import os
    root=tmp_path/'run'; root.mkdir(); (root/'logs').mkdir()
    proc=tmp_path/'proc'; proc.mkdir()
    control=root/'controls'/'drain'; old_control=tmp_path/'old-control'; old_control.mkdir()
    old_source=tmp_path/'source-release'; old_source.mkdir()
    output=root/'F/row/source'; output.mkdir(parents=True)
    train_cfg=root/'train.json'; pred_cfg=root/'pred.json'
    train_cfg.write_text(json.dumps({'row_id':'row','output_root':str(output)}),encoding='utf8')
    pred_cfg.write_text(json.dumps({'row_id':'row','source_output':str(output)}),encoding='utf8')
    jobs=[{'id':'train__row','kind':'train','config':str(train_cfg),'completion':str(output/'complete.json')},
          {'id':'predict__row','kind':'predict','config':str(pred_cfg),'dependencies':[str(output/'complete.json')]}]
    source={'row':{'row':{'row_id':'row'},'source_output':str(output),'predictions':'pred','scores':'score'}}
    (root/'jobs.json').write_text(json.dumps(jobs),encoding='utf8'); (root/'sources.json').write_text(json.dumps(source),encoding='utf8')
    prior={'status':'RUNNING','pid':100,'jobs':{'train__row':{'status':'RUNNING','pid':200,'gpu':0},'predict__row':{'status':'QUEUED'}}}
    (old_control/'pipeline_state.json').write_text(json.dumps(prior),encoding='utf8')
    rt={'run_root':str(root),'control_root':str(control),'source_jobs':str(root/'jobs.json'),
        'source_control_state':str(old_control/'pipeline_state.json'),'source_control_release':str(old_control),
        'source_control_module':'old.control','source_release':str(old_source),'worker_module':'worker.runner',
        'python':'python','launch_owner':'test','code_commit':'test','source_model_commit':'prior','gpu7_limit':1}
    for pid,argv in ((100,['python','-u','-m','old.control','--config',str(old_control)+'/runtime.json']),
                     (200,gpu01_drain.command(rt,jobs[0]))):
        p=proc/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in argv)+b'\0')
    observed={'train__row':dict(status='RUNNING',pid=200,gpu=0,cwd=str(old_source),argv=gpu01_drain.command(rt,jobs[0]),log=str(root/'logs/train__row.log'))}
    live={100,200};signals=[];RealPath=pathlib.Path
    monkeypatch.setattr(gpu01_drain,'Path',lambda x:proc if str(x)=='/proc' else RealPath(x))
    monkeypatch.setattr(gpu01_drain,'alive',lambda pid:pid in live)
    monkeypatch.setattr(gpu01_drain,'current_workers',lambda rt,jobs:observed if 200 in live else {})
    monkeypatch.setattr(gpu01_drain,'done',lambda j:RealPath(j['completion']).exists() if 'completion' in j else False)
    monkeypatch.setattr(os,'readlink',lambda p: str(old_control) if '100' in str(p) else str(old_source))
    monkeypatch.setattr(os,'getpgid',lambda pid:pid,raising=False);monkeypatch.setattr(os,'getsid',lambda pid:pid,raising=False)
    def stop(pid,sig): signals.append(pid);live.discard(pid)
    monkeypatch.setattr(os,'kill',stop);monkeypatch.setattr(os,'killpg',stop,raising=False)
    monkeypatch.setattr(gpu01_drain.time,'sleep',lambda n:None)
    monkeypatch.setattr(gpu01_drain,'loop',lambda rt,control,state,jobs,failures:(state,jobs))
    state,new_jobs=gpu01_drain.takeover(rt)
    assert signals==[100,200]
    assert state['jobs']['train__row--gpu01-r01']['status']=='QUEUED'
    assert len(json.loads((control/'jobs.json').read_text(encoding='utf8')))==2
    assert json.loads((control/'interrupt-000.json').read_text(encoding='utf8'))['status']=='REQUEUED'
    new_sources=json.loads((root/'sources.json').read_text(encoding='utf8'))
    assert new_sources['row']['source_output'].endswith('source-gpu01-r01')
    assert output.is_dir()
