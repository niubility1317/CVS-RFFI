import json
from types import SimpleNamespace
import pytest
from experiments.pamf_phase3_restoration import coordinate_remaining_dispatch as m


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def prefix(root):
    row=root/'rows'/m.ARM
    for stage in ('train','cross_rx'):put(root/('terminal-'+m.ARM+'-'+stage+'.json'),dict(exit_code=0))
    put(row/'training/complete.json',dict(status='SOURCE_TRAINED',code_commit=m.SCIENCE_COMMIT,epoch=30,updates=7410,F0_identity={'fixed':1}))
    (row/'training/final.pt').write_bytes(b'preserved')
    put(row/'prediction/cross_rx/complete.json',dict(status='PREDICTIONS_COMPLETE',run_id=m.RUN,arm=m.ARM,task='cross_rx',count=126000,
        code_commit=m.SCIENCE_COMMIT,views=m.VIEWS,readouts=['raw','DURA_v1','candidate'],truth_read=False,query_fit=False,model_state_updates=0,F0_identity={'fixed':1}))
    (row/'prediction/cross_rx/predictions.npz').write_bytes(b'preserved')
    return row


def test_completed_prefix_preserved_and_only_remaining_commands(tmp_path):
    row=prefix(tmp_path);m.verify_remaining(tmp_path)
    args=SimpleNamespace(root=tmp_path,legacy_release=tmp_path/'original',cache=tmp_path/'cache')
    command=m.stage_command(args,'cross_both')
    assert command[command.index('--commit')+1]==m.SCIENCE_COMMIT and command[-2:]==['--task','cross_both']
    assert '--task' not in m.stage_command(args,'score')
    for stage in ('train','cross_rx'):
        with pytest.raises(ValueError):m.stage_command(args,stage)
    assert (row/'training/final.pt').read_bytes()==b'preserved'


def test_partial_next_stage_prevents_duplicate(tmp_path):
    row=prefix(tmp_path);(row/'prediction/cross_both').mkdir()
    with pytest.raises(ValueError):m.verify_remaining(tmp_path)


def test_bad_prefix_or_scientific_identity_rejected(tmp_path):
    row=prefix(tmp_path)
    p=row/'training/complete.json';v=json.loads(p.read_text());v['epoch']=29;put(p,v)
    with pytest.raises(ValueError):m.verify_remaining(tmp_path)
    v['epoch']=30;put(p,v)
    p=row/'prediction/cross_rx/complete.json';v=json.loads(p.read_text());v['code_commit']='wrong';put(p,v)
    with pytest.raises(ValueError):m.verify_remaining(tmp_path)
