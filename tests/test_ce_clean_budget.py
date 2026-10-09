import copy
import pytest
from experiments.cvs_ce_batch_sweep.evaluate_clean import validate_budget,validate_freeze,ROW_IDS,RUN,SELECTION

@pytest.mark.parametrize('batch,steps',[(256,25),(512,13),(1024,7),(2048,4)])
def test_exact_singlepass_budget_accepts_and_old_or_wrong_row_rejects(batch,steps):
    c=dict(batch_size=batch,steps_per_epoch=steps)
    done=dict(status='SOURCE_TRAINED',epoch=200,config=c,logged_steps=200*steps,optimizer_steps=200*steps,target_access=False,target_evaluated=False,checkpoint_sources=[])
    assert validate_budget(c,done)==200*steps
    for change in [dict(logged_steps=44400,optimizer_steps=44400),dict(optimizer_steps=200*steps-1),dict(target_access=True),dict(epoch=199),dict(config={**c,'batch_size':1}),dict(checkpoint_sources=['unknown'])]:
        with pytest.raises(ValueError):validate_budget(c,{**done,**change})

def test_frozen_four_row_matrix():
    rows=[dict(row_id=r) for r in ROW_IDS]
    v=dict(run_id=RUN,selection=SELECTION,rows=rows,target_access=False)
    assert validate_freeze(v,rows)==rows
    for change in [dict(rows=rows[:-1]),dict(target_access=True),dict(selection='selected_by_target')]:
        with pytest.raises(ValueError):validate_freeze({**v,**change},rows)

def test_preflight_uses_per_row_provenance_after_design_lookup(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from contextlib import nullcontext
    import torch
    from experiments.cvs_phase1_repair import evaluate as e
    from experiments.cvs_phase1_stack import runtime,recover
    from experiments.cvs_ce_batch_sweep import evaluate_clean as clean
    c=dict(row_id=ROW_IDS[0],batch_size=256,steps_per_epoch=25,output_root='/source')
    done=dict(status='SOURCE_TRAINED',epoch=200,config=c,logged_steps=5000,optimizer_steps=5000,target_access=False,target_evaluated=False,checkpoint_sources=[])
    freeze=dict(run_id=RUN,selection=SELECTION,rows=[c],target_access=False)
    def read(path):
        name=Path(path).name
        if name=='completion.json':return done
        if name=='initialization.json':return dict(scratch_only=True,ancestors=[],checkpoint_sources=[],target_contact=False,source_roles='EXACT_MATCH')
        if name=='source_matrix_frozen.json':return freeze
        return {}
    for name in ['source_provenance','design','BASE','RUN','ROOT','VIEWS','snapshot','write']:
        monkeypatch.setattr(e,name,getattr(e,name))
    monkeypatch.setattr(clean,'ROW_IDS',(ROW_IDS[0],))
    monkeypatch.setattr(e,'read',read)
    monkeypatch.setattr(e,'checkpoint_provenance',lambda d,c,ck:None)
    monkeypatch.setattr(recover,'verify_contract',lambda a,b:None)
    monkeypatch.setattr(runtime,'installed',lambda c:nullcontext())
    monkeypatch.setattr(torch,'load',lambda *a,**kw:{})
    def forbidden_old(*args):raise AssertionError('Old44400 budget checker called')
    d=SimpleNamespace(SOURCE='/contract',require_budget=forbidden_old)
    clean.configure(e,d,[c])
    assert e.VIEWS==('clean',)
    assert e.preflight()['models']==1
    done['optimizer_steps']=44400
    with pytest.raises(ValueError):e.preflight()
