import pytest


def test_match_templates_uses_exact_profile_capsule_and_seed():
    from experiments.pamf_phase2_practical.clean_core_native_da import match_templates
    target=dict(profile='dadda_sda',capsule='capsules/P01/MIX01/1-1',seed=2026092701,k=10)
    job=dict(profile='dadda_sda',capsule=target['capsule'],seed=target['seed'],points=[[5,20]])
    entry=dict(group='native_da',id='pilot',runtime='/runtime')
    assert match_templates([target],[(entry,job)])==[(target,entry,job)]
    with pytest.raises(ValueError,match='template'):
        match_templates([target],[(entry,dict(job,seed=2026092702))])
    with pytest.raises(ValueError,match='template'):
        match_templates([target],[(entry,job),(dict(entry,id='other'),job)])


def test_prepare_writes_only_the_missing_k10_baselines(tmp_path):
    import json
    from experiments.pamf_phase2_practical.clean_core_native_da import prepare
    def write(path,data):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data));return str(path)
    parent=tmp_path/'data';parent.mkdir();clean=tmp_path/'clean'
    write(clean/'completion.json',dict(status='CLEAN_PAIRED_COMPLETE',parent_task_root=str(parent)))
    source=tmp_path/'source';source.mkdir();cache=tmp_path/'cache';cache.mkdir()
    phase1=tmp_path/'phase1';phase1.mkdir();release=tmp_path/'release';release.mkdir()
    contract=write(tmp_path/'contract.json',{});seed=2026092701
    profiles=('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft','tweak_m1')
    receivers=('1-1','14-7','2-1','20-1','7-14');targets=[];entries=[];missing={}
    for profile in profiles:
        model=source/'source'/f'{profile}-s{seed}';model.mkdir(parents=True);(model/'source.pt').write_bytes(b'x')
        runtime=dict(clean_root=str(clean),data_root=str(parent),source_contract=contract,source_cache=str(cache),
            phase1_release=str(phase1),run_root=str(source),release_root=str(release),
            additional_sources={'tweak_m1':{str(seed):str(model)}})
        rt=write(tmp_path/'old'/f'{profile}-rt.json',runtime);missing['native_da/'+profile]=[]
        for rx in receivers:
            cap=f'capsules/P01/MIX01/{rx}';targets.append(dict(profile=profile,capsule=cap,seed=seed,k=10))
            for stage in ('A0','B'):missing['native_da/'+profile].append(dict(capsule=cap,seed=seed,stage=stage,k=10,n=0))
            job=dict(profile=profile,capsule=cap,seed=seed,points=[[5,20]],output=str(tmp_path/'old-output'))
            entries.append(dict(id=f'{profile}-{rx}',group='native_da',runtime=rt,
                                job=write(tmp_path/'old'/f'{profile}-{rx}.json',job)))
    parent_file=write(tmp_path/'preparations/pilot/scheduler.json',dict(jobs=entries))
    targets_file=write(tmp_path/'targets.json',targets)
    missing_file=write(tmp_path/'missing.json',dict(scope='MIX01-08 K5/10 N5/10',missing=missing))
    with pytest.raises(ValueError,match='discoverable'):
        prepare(parent_file,targets_file,missing_file,tmp_path/'preparations/bad-name',tmp_path/'run',release)
    with pytest.raises(ValueError,match='discoverable'):
        prepare(parent_file,targets_file,missing_file,tmp_path/'preparations/phase2.clean',tmp_path/'run',release)
    with pytest.raises(ValueError,match='discoverable'):
        prepare(parent_file,targets_file,missing_file,tmp_path/'preparations/20261112-phase2-clean-pilot-r01',tmp_path/'run',release)
    plan=tmp_path/'preparations/20261116-phase2-clean-core-native-da-r01'
    result=prepare(parent_file,targets_file,missing_file,plan,tmp_path/'run',release)
    assert result['jobs']==30 and not (tmp_path/'run').exists()
    config=json.loads((plan/'scheduler.json').read_text())
    assert config['phase']=='core_native_da'
    assert all(json.loads(open(e['job']).read())['points']==[[10,5]] for e in config['jobs'])
