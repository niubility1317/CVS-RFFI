import json
from pathlib import Path
import pytest
from experiments.pamf_channel_pl.references import resolve_reference_score_paths,repair,SCIENCE_RUN,OWNER


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def job(root,rid,folder,name=None,seed=2026092701,status='SCORED_COMPLETE'):
    out=root/folder;config=root/((name or folder)+'.json');protocol=rid.rsplit('-s',1)[0].rsplit('-',1)[0]
    write(config,dict(row_id=rid,model_seed=seed,protocol_id=protocol,run_id='old',output_root=str(out)))
    write(out/'complete.json',dict(status=status,row_id=rid,run_id='old'))
    write(out/'scores.json',dict(status='SCORED',truth_last=True,row_id=rid,run_id='old',model_seed=seed))
    return dict(kind='score',config=str(config),completion=str(out/'complete.json'))


def test_resolves_recovered_score_from_recorded_job_instead_of_canonical_guess(tmp_path):
    rid='F0-supervised-s2026092701';j=job(tmp_path,rid,'score_recovery/row','recovered')
    path=tmp_path/'jobs.json';write(path,[j])
    assert resolve_reference_score_paths(path,{rid})[rid]==str(tmp_path/'score_recovery/row/scores.json')


def test_does_not_choose_between_two_completed_scores_by_their_values(tmp_path):
    rid='F0-supervised-s2026092701';path=tmp_path/'jobs.json'
    write(path,[job(tmp_path,rid,'a'),job(tmp_path,rid,'b')])
    with pytest.raises(ValueError,match='Ambiguous'):resolve_reference_score_paths(path,{rid})


@pytest.mark.parametrize('bad',['missing','seed'])
def test_rejects_missing_or_wrong_seed_reference(tmp_path,bad):
    rid='F0-supervised-s2026092701';j=job(tmp_path,rid,'a',seed=7 if bad=='seed' else 2026092701)
    if bad=='missing':Path(j['completion']).unlink()
    path=tmp_path/'jobs.json';write(path,[j])
    with pytest.raises(ValueError):resolve_reference_score_paths(path,{rid})


def test_report_path_repair_preserves_original_and_is_idempotent(tmp_path):
    runs=tmp_path/'runs';root=runs/SCIENCE_RUN;prior=runs/'old';newrid='F0-ChannelPL-s2026092701'
    refs={newrid:{}};jobs=[]
    for method in ('supervised','ordinary_PL','REPL'):
        rid='F0-'+method+'-s2026092701';j=job(prior,rid,method);jobs.append(j)
        refs[newrid][method]=str(prior/'missing'/method/'scores.json') if method=='supervised' else str(prior/method/'scores.json')
    write(prior/'jobs.json',jobs);write(root/'reference_scores.json',refs)
    write(root/'pipeline_state.json',dict(code_commit='science',launch_owner=OWNER))
    preview=repair(root,prior/'jobs.json','science',apply=False)
    assert len(preview['changes'])==1 and json.loads((root/'reference_scores.json').read_text())==refs
    actual=repair(root,prior/'jobs.json','science',apply=True)
    assert actual['status']=='VERIFIED_REPAIRED' and json.loads((root/'reference_scores_before_path_repair.json').read_text())==refs
    assert repair(root,prior/'jobs.json','science',apply=True)['status']=='VERIFIED_ALREADY_CORRECT'
