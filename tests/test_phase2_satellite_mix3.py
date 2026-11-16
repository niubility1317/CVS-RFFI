import json

import pytest

from experiments.pamf_phase2_practical import satellite_mix3 as m


def test_revised_matrix_excludes_existing_and_keeps_single_owner(tmp_path):
    data=tmp_path/'data'; data.mkdir()
    contract=tmp_path/'contract.json'; contract.write_text('{}')
    cache=tmp_path/'cache'; cache.mkdir()
    release=tmp_path/'release'; release.mkdir()
    rt={'data_root':str(data),'source_contract':str(contract),'source_cache':str(cache),'clean_root':'forbidden'}
    sources={}
    for key in ('da','hybrid',*m.NATIVE):
        path=tmp_path/(key+'.json');path.write_text(json.dumps(rt));sources[key]=str(path)
    caps=[f'capsules/{pool}/MIX{scene:02}/{rx}' for pool in ('P01','P02','P03')
          for scene in (1,2,3) for rx in m.RECEIVERS]
    present=[dict(capsule=cap,seed=seed,k=k) for cap in caps for seed in m.SEEDS[:3]
             for k in ((5,10) if '/MIX01/' in cap else (5,))]
    a0={'status':'VERIFIED','issues':[],'existing':{p:present for p in ('lora_triplet_knn','mopc_hr')}}
    cap=caps[7]
    profiles={'protonet_cda':{'missing_pairs':[]}}
    for name in ('mrior_sda','twostage','poster_ft'):
        profiles[name]={'missing_pairs':[dict(capsule=cap,seed=m.SEEDS[0],k=10)]}
    inv={'status':'VERIFIED','issues':[],'profiles':profiles}
    config=m.prepare(inv,a0,sources,tmp_path/'run',release,tmp_path/'config'/'scheduler.json',tmp_path/'old')
    jobs=config['jobs']
    assert len(jobs)==3+189*3+189+189*5
    assert config['allowed_gpus']==list(range(8)) and config['max_per_gpu']==6
    assert len({j['id'] for j in jobs})==len(jobs)
    assert all(j['capsule'].split('/')[2] in ('MIX01','MIX02','MIX03') and j['seed'] in m.SEEDS[:3] for j in jobs)
    assert all(j['points']==m.POINTS and j['save_B_for_mopc'] for j in jobs if j['group']=='hybrid')
    assert all(not json.loads(open(path).read()).get('clean_root') for path in config['runtimes'].values())


def test_unverified_reuse_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        m.prepare({'status':'INCOMPLETE'}, {}, {}, tmp_path/'run', tmp_path, tmp_path/'p'/'c', tmp_path/'old')
