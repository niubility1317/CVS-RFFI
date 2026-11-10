import json
from pathlib import Path
from experiments.pamf_split_spectrum_paper.recover_567 import plan_recovery, remap, source_contract_matches


def test_recovery_preserves_completed_and_reconnects_failed_checkpoint(tmp_path):
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir()
    jobs=[];state={'jobs':{}}
    for kind,status in [('train','FAILED'),('predict','BLOCKED_TECHNICAL_DEPENDENCY'),('score','BLOCKED_TECHNICAL_DEPENDENCY')]:
        output=old/kind;cfg=old/(kind+'.json');cfg.write_text(json.dumps(dict(output_root=str(output),
            source_output=str(old/'train'),prediction_output=str(old/'predict'))))
        jid=kind+'__row';deps=[] if kind=='train' else [str(old/('train' if kind=='predict' else 'predict')/'complete.json')]
        jobs.append(dict(id=jid,kind=kind,family='architecture',config=str(cfg),completion=str(output/'complete.json'),dependencies=deps))
        state['jobs'][jid]=dict(status=status,failure="AttributeError: 'CVSincNet' object has no attribute '_factorial_aux'")
    result,sources=plan_recovery(old,new,jobs,state,{'row':dict(source_output=str(old/'train'),scores=str(old/'score'))},'new-commit')
    assert result[0]['_new_config']['recover_ce_from']==str(old/'train')
    assert result[1]['dependencies']==[str(new/'train'/'complete.json')]
    assert result[1]['_new_config']['source_output']==str(new/'train')
    assert result[2]['dependencies']==[str(new/'predict'/'complete.json')]
    assert sources['row']['scores']==str(new/'score')
    assert remap(str(old/'train-other'),{str(old/'train'):str(new/'train')})==str(old/'train-other')
    (new/'configs').mkdir(parents=True)
    for job in result:
        Path(job['config']).write_text(json.dumps(job.pop('_new_config')))
    state['jobs']['train__row']['failure']='ValueError: Saved training evidence is incomplete or incompatible'
    again,_=plan_recovery(new,tmp_path/'r02',result,state,sources,'later-commit')
    assert again[0]['_new_config']['recover_ce_from']==str(old/'train')
    assert again[0]['_new_config']['recovery_commit']=='later-commit'


def test_source_contract_compares_real_roles_and_derived_representation():
    original={'role_ids':{'L_s':[1,2]},'source_rxs':[1,3], 'split_seed':392005}
    saved=dict(original, physical_roles='EXACT_MATCH',dataset_path='/dataset',classes=list('abcdef'),
               equalized=1,out_len=256,normalize=True)
    assert source_contract_matches(saved,original,{'dataset':'/dataset'})
    assert not source_contract_matches(dict(saved,role_ids={'L_s':[3]}),original,{'dataset':'/dataset'})
    assert not source_contract_matches(dict(saved,normalize=False),original,{'dataset':'/dataset'})
