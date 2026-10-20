import copy
from types import SimpleNamespace
import pytest
from experiments.pamf_positive_unet_restore import design as d
from experiments.pamf_positive_unet_restore.runtime import source_statistics,gain,audit_contract
from experiments.pamf_positive_unet_restore.runner import configuration,command
from experiments.pamf_uncertainty_restore.evaluation import VIEWS
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN
from experiments.pamf_restoration.data import write_json


def source():
    return dict(status='ANALYZED',role='source',count=1800,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,
        training_F_access=False,head_fit_updates=0,target_access=False,
        records=[dict(readout='restored',dimension='overall',view=v,count=1800,accuracy=90.) for v in VIEWS])


def test_fixed_subset_is_later_unet_only_and_prioritizes_R23():
    assert len(d.ROWS)==12 and len(d.NEW_ROWS)==8
    assert d.NEW_ROWS[:2]==('r23_prior_only','r23_joint_coarse')
    assert all(s['architecture']=='U-Net restoration prior' for s in d.ROWS)
    assert not any('diffusion' in k or 'flow' in k or 'tcn' in k or 'attention' in k for k in d.BY_ID)
    for spec in d.ROWS:
        if spec['source_control'] is not None:
            actual=gain(spec['source_statistics'],d.BY_ID[spec['source_control']]['source_statistics'])
            assert actual['mean6_gain_pp']>0


@pytest.mark.parametrize('key,value',[('role','target'),('target_access',True),('training_F_access',True),('head_fit_updates',1),('F_seed',2026092702),('all_predictions_before_truth',False)])
def test_wrong_source_evidence_is_rejected(key,value):
    s=source();s[key]=value
    with pytest.raises(ValueError):source_statistics(s)


def test_missing_scene_blocks_source_selection():
    s=source();s['records'].pop()
    with pytest.raises(ValueError):source_statistics(s)


@pytest.mark.parametrize('mean6,clean',[(90.,98.),(89.,98.),(91.,97.)])
def test_zero_negative_gain_or_clean_damage_is_rejected(mean6,clean):
    with pytest.raises(ValueError):gain(dict(mean6=mean6,clean=clean,lowurban=80.,midurban=90.),dict(mean6=90.,clean=98.,lowurban=80.,midurban=90.))


def test_mixed_scenes_are_reported_without_faking_all_scene_wins():
    result=gain(dict(mean6=91.,clean=98.,lowurban=79.,midurban=92.),dict(mean6=90.,clean=98.,lowurban=80.,midurban=90.))
    assert result['mean6_gain_pp']==1 and result['lowurban_gain_pp']==-1


def test_old_all_completed_authorization_cannot_open_new_subset(tmp_path):
    from experiments.pamf_backfill_restore.runner import configuration as old
    write_json(tmp_path/'resolved_config.json',old('a'))
    with pytest.raises(ValueError):audit_contract(SimpleNamespace(output=tmp_path,expected_commit='a'))


def test_unfrozen_source_selection_cannot_open_target(tmp_path):
    write_json(tmp_path/'resolved_config.json',configuration('a'));write_json(tmp_path/'source_positive_selection.json',dict(status='missing'))
    with pytest.raises(ValueError):audit_contract(SimpleNamespace(output=tmp_path,expected_commit='a'))


def test_new_commands_use_new_owner_module_and_original_zero_training_config(tmp_path):
    from experiments.pamf_positive_unet_restore.runner import PATH_ARGS
    args=SimpleNamespace(**{k:tmp_path/k for k in PATH_ARGS},expected_commit='a')
    cmd=command(args,'worker','r23_prior_only')
    assert cmd[cmd.index('-m')+1]=='experiments.pamf_positive_unet_restore.runner'
    assert configuration('a')['new_training_updates']==0
