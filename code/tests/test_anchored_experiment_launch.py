import importlib.util,json,sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_core90_anchored_experiment.py'
spec=importlib.util.spec_from_file_location('anchored_launch',SCRIPT)
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)


def test_source_materialization_uses_frozen_ids_without_target_indexing():
    class Forbidden:
        def __getitem__(self,key):raise AssertionError('target IQ touched')
    array=np.random.default_rng(9).normal(size=(4,256,2)).astype('float32')
    ds=dict(tx_list=['a'],rx_list=['source','target'],capture_date_list=['d'],equalized_list=[0,1],
            data=[[[[array,array]],Forbidden()]])
    contract=dict(tx_mapping=['a'],equalized=1,source_receivers=[0],source_days=[0],roles={'L_s':['0:0:0:2'],'V':['0:0:0:3'],'target':['0:1:0:0']})
    data=launch.source_dataset(ds,contract,'L_s',256)
    assert len(data)==1 and data[0][1]==0 and data.index[0].sig_i==2 and data.index[0].eq_i==1
    with pytest.raises(ValueError):launch.source_dataset(ds,contract,'target',256)
    contract['roles']['V']=['0:1:0:0']
    with pytest.raises(ValueError):launch.source_dataset(ds,contract,'V',256)


@pytest.mark.parametrize('active',[True,False])
def test_coordinator_single_seed_matrix_has_no_target_or_duplicate_fits(tmp_path,monkeypatch,active):
    root=tmp_path/'run';logs=tmp_path/'logs';calls=[]
    plan=dict(logs=str(logs),seed=392005,gpus=[0,1,2,3,5],threads=2,config='config',ground='ground',contract='contract')
    def child(plan,root,label,cmd,gpu):
        calls.append((label,cmd,gpu))
        if '--stage' in cmd and cmd[cmd.index('--stage')+1]=='fuse' and cmd[cmd.index('--candidate')+1]=='A6':
            out=Path(cmd[cmd.index('--output')+1]);out.mkdir(parents=True)
            (out/'activation.json').write_text(json.dumps(dict(status='SOURCE_NESTED_EVALUATED' if active else 'NOT_ACTIVATED_NO_SOURCE_UTILITY')))
        return label,None
    monkeypatch.setattr(launch,'child',child);monkeypatch.setattr(launch,'wait_children',lambda jobs:None)
    launch.coordinate(plan,root)
    stages=[(cmd[cmd.index('--stage')+1],cmd[cmd.index('--candidate')+1]) for _,cmd,_ in calls if '--stage' in cmd]
    assert [c for s,c in stages if s=='oof']==list(launch.EXPERTS)
    assert [c for s,c in stages if s=='fit']==['A0','A1','P1']
    assert [c for s,c in stages if s=='fuse']==[*launch.FIXED,'A6']
    assert ('calibrate','A6') in stages if active else ('calibrate','A6') not in stages
    assert len([c for s,c in stages if s=='export'])==(14 if active else 13)
    assert all(s in {'cache','fit','oof','fuse','calibrate','export','profile'} for s,c in stages)
    assert len({g for label,cmd,g in calls if label.endswith('_oof')})==5
    state=json.loads((root/'status.json').read_text())
    assert state['stage']=='SOURCE_SYSTEM_FROZEN' and not state['target_access']
    with pytest.raises(FileExistsError):launch.coordinate(plan,root)


def test_stage_commands_keep_seed_and_oof_path(tmp_path):
    plan=dict(seed=392005,config='cfg',ground='ground',contract='contract',threads=4)
    cmd=launch.stage_command(plan,tmp_path,'calibrate','A4',tmp_path/'cal',cache=tmp_path/'V',input=tmp_path/'oof/A4/system_state.pt')
    assert cmd[cmd.index('--seed')+1]=='392005'
    assert cmd[cmd.index('--input')+1]==str(tmp_path/'oof/A4/system_state.pt')
    assert '--source' not in cmd


def test_dispatch_payload_is_source_only_and_collision_protected():
    path=Path(__file__).resolve().parents[2]/'tools/dispatch_core90_anchored_392005.py'
    spec=importlib.util.spec_from_file_location('dispatch',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    code=module.payload('/home/szu2070436088/2510044040/CV-SincNet/releases/test')
    compile(code,'<dispatch>','exec')
    assert "pidfile.open('x'" in code and "log.open('x'" in code
    assert "'--gpus','0,1,2,3,5'" in code and 'target_predictions' not in code
