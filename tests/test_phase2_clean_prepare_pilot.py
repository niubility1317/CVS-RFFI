import json

from experiments.pamf_phase2_practical.clean_prepare_pilot import prepare
from experiments.pamf_phase2_practical.clean_scheduler import validate_plan


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def test_prepare_pilot_emits_bound_queue_and_explicit_blockers(tmp_path):
    data=tmp_path/'data';data.mkdir();clean=tmp_path/'clean'
    put(clean/'completion.json',{'status':'CLEAN_PAIRED_COMPLETE','parent_task_root':str(data)})
    source=tmp_path/'source';seed=2026092701
    for profile in ('protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft','lora_triplet_knn'):
        p=source/'source'/f'{profile}-s{seed}'/'source.pt';p.parent.mkdir(parents=True);p.write_bytes(b'x')
    extras={}
    for profile in ('tweak_m1','issl_fixed_v1'):
        p=tmp_path/profile;p.mkdir();(p/'source.pt').write_bytes(b'x');extras[profile]={str(seed):str(p)}
    cache=tmp_path/'cache';cache.mkdir();phase1=tmp_path/'phase1';phase1.mkdir();release=tmp_path/'release';release.mkdir()
    contract=put(tmp_path/'contract.json',{});pamf_src=tmp_path/'pamf';pamf_src.mkdir();(pamf_src/'final_ssdg.pth').write_bytes(b'x')
    dura=tmp_path/'dura.pt';dura.write_bytes(b'x')
    scalr=tmp_path/'scalr'/f's{seed}';scalr.mkdir(parents=True)
    (scalr/'source_selection.json').write_text('{}');(scalr/'source_orbit_cca.npz').write_bytes(b'x')
    common=dict(data_root=str(data),source_contract=str(contract),source_cache=str(cache),phase1_release=str(phase1))
    pamf=put(tmp_path/'pamf.json',dict(common,arm='R42',kernel_method='adaptive_residual_ensemble',
             PAMF_sources={str(seed):str(pamf_src)},DURA={'checkpoint':str(dura)}))
    group1=put(tmp_path/'group1.json',common);additional=put(tmp_path/'additional.json',{'additional_sources':extras})
    summary=prepare(tmp_path/'plan',tmp_path/'run',release,clean,pamf,group1,additional,source,scalr.parent,
        capsules=['capsules/P01/MIX01/1-1'],seed=seed,allowed_gpus=[0,1])
    assert summary['jobs']==15 and len(summary['blocked'])==8 and not summary['formal_experiment_started']
    assert {r['profile'] for r in summary['blocked'] if r['group']=='hybrid'}=={
        'protonet_cda','mrior_sda','dadda_sda','twostage','poster_ft','tweak_m1'}
    config=json.loads((tmp_path/'plan/scheduler.json').read_text())
    assert validate_plan(config)['jobs']==15
    assert not (tmp_path/'run').exists()
    repaired=tmp_path/'repaired'
    folder=repaired/'source'/f'orthogonal-s{seed}'
    put(folder/'recipe.json',{'schema':'group1_source_orthogonal_tempered_PL_v2'})
    (folder/'source.pt').write_bytes(b'x')
    summary=prepare(tmp_path/'ready-plan',tmp_path/'ready-run',release,clean,pamf,group1,additional,source,scalr.parent,
        capsules=['capsules/P01/MIX01/1-1'],seed=seed,allowed_gpus=[0,1],orthogonal_source_root=repaired)
    assert summary['jobs']==16 and len(summary['blocked'])==7
    config=json.loads((tmp_path/'ready-plan/scheduler.json').read_text())
    assert validate_plan(config)['jobs']==16
    orthogonal=next(entry for entry in config['jobs'] if entry['id']=='cil-orthogonal-00')
    assert json.loads(open(orthogonal['runtime']).read())['run_root']==str(repaired)
