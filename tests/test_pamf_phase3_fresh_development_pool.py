"""Synthetic three-member Full pool, privacy and physical-view boundaries."""
import importlib
import json
from pathlib import Path
import numpy as np
import pytest


def api():return importlib.import_module('experiments.pamf_phase3_open_world.fresh_development_pool')


def fixture(root,mp):
    m=api();mp.setattr(m,'U_PER_TX',2);mp.setattr(m,'U_COUNT',18);mp.setattr(m,'Q_COUNT',36);mp.setattr(m,'Q_BY_DAY',dict(zip(m.FUTURE_DAYS,(20,16))))
    full=root/'Full';full.mkdir();validation=root/'validation';validation.mkdir()
    rx=list(m.SOURCE_RX)+['1-1','14-7','2-1','20-1','7-14','7-7','8-8']+[f'r{i}' for i in range(22)];rx.insert(34,m.RX);rx+=['24-1','24-18']+[f'extra{i}' for i in range(41-len(rx)-2)]
    tx=m.OLD+m.NEW+m.LATER+[f'tx{i}' for i in range(174-10)];counts=np.zeros((4,41,174),np.int64);payload={};rows=[]
    days=[m.EARLY,'2021_03_08',*m.FUTURE_DAYS]
    for day in [m.EARLY,*m.FUTURE_DAYS]:
        arrays=[]
        for j,t in enumerate(m.OLD+m.NEW+m.LATER):
            n=4 if day==m.EARLY else 0 if day==m.FUTURE_DAYS[1] and t in ('20-19','8-20') else 2
            counts[days.index(day),34,j]=n;raw=np.random.default_rng(j+days.index(day)*30).normal(0,4,(n,256,2)).astype(np.float64);arrays.append(raw)
        payload[day]=dict(node_list=tx[:10],data=arrays);rows.append(dict(date=day,receiver=m.RX,status='PASS',archive=day+'.zip',member=day+'.pkl'))
    def write(path,obj):path.write_text(json.dumps(obj),encoding='utf-8')
    write(validation/'summary.json',dict(status='VERIFIED',dataset_directory=str(full),members_checked=158,expected_nonempty_RX_date_members=158,count_mismatch_cells=0,errors=[]))
    write(validation/'counts.json',dict(dates=days,receivers=rx,transmitters=tx,records=counts.tolist()))
    (validation/'members.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    compact=rx[:34]+[rx[35]];compact=list(dict.fromkeys(compact));assert len(compact)==35
    audit=dict(schema='pamf_phase3_fresh_development_metadata_audit_v1',scope=dict(receiver=m.RX,Full_receiver_index=34,early_day=m.EARLY,future_days=list(m.FUTURE_DAYS),registered_classes=m.OLD,fresh_discovery_candidates=m.NEW,future_only_unknown_candidates=m.LATER),
        fresh_TX_exclusions=dict(groups=dict(original_known_6=m.OLD,other=[f'excluded{i}' for i in range(28)]),union_count=34,fresh_four=m.NEW+m.LATER,intersection=[],disjoint_confirmed=True),
        receiver_role_exclusions=dict(complete_compact_RX_union=compact,complete_compact_RX_union_count=35,compact_RX_lists={k:compact for k in ('ManyRx.pkl','ManySig.pkl','ManyTx.pkl','SingleDay.pkl')},current_source_receivers=m.SOURCE_RX,
            current_Phase2_target_receivers=['1-1','14-7','2-1','20-1','7-14','7-7','8-8'],old_Phase3_development_receiver='24-1',formal_Phase3_reserved_receiver='24-18',examined_actual_canonical_role_indices=[dict(receiver_names=['24-1']),dict(receiver_names=['24-18'])]))
    write(root/'audit.json',audit);mixture=json.loads(Path('configs/pamf_phase2_practical_mixture_20261028.json').read_text(encoding='utf-8-sig'));write(root/'mixture.json',mixture)
    c=dict(run_id=m.RUN_ID,code_commit='c'*40,full_root=str(full),validation_root=str(validation),metadata_audit=str(root/'audit.json'),mixture_config_path=str(root/'mixture.json'))
    events=[];observe=m._observe;calls=[]
    def channel(raw,**kw):
        if kw['physical_id'].startswith('synthetic-'):events.append('smoke');return observe(raw,**kw)
        assert raw.dtype==np.float64 and raw.shape==(2,256) and np.std(raw)>1
        calls.append(kw);return np.full((2,256),2.,np.float32),dict(scenario='practical_low_urban',overlay_seed=1,input_rms=4.,geometry={})
    def member(config,locator):
        assert events[:2]==['smoke','smoke'];events.append(locator['date']);return payload[locator['date']]
    mp.setattr(m,'_observe',channel);mp.setattr(m,'_read_member',member)
    return m,c,payload,events,calls


def test_three_members_one_normalization_private_truth_and_paired_future(tmp_path,monkeypatch):
    m,c,payload,events,calls=fixture(tmp_path,monkeypatch);out=tmp_path/'pool';done=m.build(c,out)
    assert events==['smoke','smoke',m.EARLY,*m.FUTURE_DAYS] and len(calls)==54
    assert done['unique_canonical_count']==54 and done['discovery_count']==18 and done['evaluation_count']==36 and not done['global_history_unused_verified']
    loaded={}
    for role in ('discovery','future_star','future_clean'):
        h=done[role];manifest,ids,iq=m.load_public(h['public_root'],h['capsule_id'],h['split_id'],h['count']);loaded[role]=(ids.copy(),iq.copy());iq._mmap.close()
        assert set(Path(h['public_root']).iterdir())=={Path(h['public_root'])/n for n in ('manifest.json','ids.npy','received.npy')}
        assert not set(manifest)&{'tx','y','day','unknown_classes','unknown_count','scenario','quota'}
    np.testing.assert_array_equal(loaded['future_star'][0],loaded['future_clean'][0]);assert not set(loaded['discovery'][0])&set(loaded['future_star'][0])
    assert np.all(loaded['future_star'][1]==2)  # No output RMS normalization.
    rms=np.sqrt(np.mean(np.sum(loaded['future_clean'][1].astype(np.float64)**2,axis=1),axis=1));np.testing.assert_allclose(rms,1,atol=1e-6)
    with np.load(out/'discovery/truth/truth.npz') as a:
        assert set(a.files)=={'ids','y','tx','receiver','day','scenario'} and '7-11' not in a['tx'] and a['y'].max()==8
    with pytest.raises(FileExistsError):m.build(c,out)


def test_private_configuration_and_bad_member_index_never_complete(tmp_path,monkeypatch,subtests):
    m,c,payload,events,calls=fixture(tmp_path,monkeypatch)
    for delta in (dict(unknown_classes=['x']),dict(receiver='24-1'),dict(truth='private')):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
    payload[m.FUTURE_DAYS[0]]['data'][0]=payload[m.FUTURE_DAYS[0]]['data'][0][:1]
    with pytest.raises(ValueError,match='count'):m.build(c,tmp_path/'bad')
    assert not Path(tmp_path/'bad/complete.json').exists()

