import json
from types import SimpleNamespace
import pytest
from experiments.pamf_phase3_restoration import phase_supervised_remaining_dispatch as m


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def prefix(root):
    row = root/'rows'/m.ARM
    put(root/('terminal-'+m.ARM+'-train.json'), {'exit_code':0})
    put(row/'training/complete.json', dict(status='SOURCE_TRAINED', code_commit=m.SCIENCE_COMMIT, epoch=30, updates=7410))
    (row/'training/final.pt').write_bytes(b'preserved')
    return row


def test_completed_source_preserved_and_only_remaining_commands(tmp_path):
    row=prefix(tmp_path);m.verify_remaining(tmp_path)
    args=SimpleNamespace(root=tmp_path,legacy_release=tmp_path/'original',cache=tmp_path/'cache')
    for stage in ('cross_rx','cross_both'):
        command=m.stage_command(args,stage)
        assert command[command.index('--commit')+1]==m.SCIENCE_COMMIT and command[-2:]==['--task',stage]
    assert '--task' not in m.stage_command(args,'score')
    with pytest.raises(ValueError):m.stage_command(args,'train')
    assert (row/'training/final.pt').read_bytes()==b'preserved'


@pytest.mark.parametrize('stage',['cross_rx','cross_both'])
def test_existing_prediction_prevents_duplicate(tmp_path,stage):
    row=prefix(tmp_path);(row/'prediction'/stage).mkdir(parents=True)
    with pytest.raises(ValueError):m.verify_remaining(tmp_path)


def test_wrong_training_or_existing_owner_rejected(tmp_path):
    row=prefix(tmp_path);p=row/'training/complete.json';v=json.loads(p.read_text());v['epoch']=29;put(p,v)
    with pytest.raises(ValueError):m.verify_remaining(tmp_path)
    v['epoch']=30;put(p,v);put(tmp_path/'resource_controller.json',{'pid':1})
    with pytest.raises(ValueError):m.verify_remaining(tmp_path)
