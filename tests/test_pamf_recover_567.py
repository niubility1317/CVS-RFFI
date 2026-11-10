import json
from pathlib import Path
from experiments.pamf_split_spectrum_paper.recover_567 import plan_recovery, remap


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
