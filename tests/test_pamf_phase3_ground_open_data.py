"""Synthetic source-only clean/G(clean) data and blinded paired query contract."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('experiments.pamf_phase3_open_world.ground_open_data')


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def validation_rows():
    groups = [(y, rx, day) for y in range(6) for rx in (1, 3, 4, 6, 8) for day in (1, 2, 3) for _ in range(2)]
    return (np.asarray(['v%04d' % i for i in range(len(groups))]),
            *[np.asarray([g[j] for g in groups], np.int64) for j in range(3)])


def test_split_is_balanced_disjoint_and_independent_of_input_order(monkeypatch):
    m = api(); monkeypatch.setattr(m, 'CELL_SIZE', 2)
    ids, y, rx, day = validation_rows()
    cal, test = m._split_validation(ids, y, rx, day, m.SPLIT_SEED)
    assert len(cal) == len(test) == 90 and not set(cal) & set(test)
    assert set(cal) | set(test) == set(range(180))
    for c in range(6):
        for r in m.RECEIVERS:
            for d in (1, 2, 3):
                assert np.count_nonzero((y[cal] == c) & (rx[cal] == r) & (day[cal] == d)) == 1
    p = np.random.default_rng(9).permutation(len(ids))
    a, b = m._split_validation(ids[p], y[p], rx[p], day[p], m.SPLIT_SEED)
    assert set(ids[cal]) == set(ids[p][a]) and set(ids[test]) == set(ids[p][b])
    with pytest.raises(ValueError):
        m._split_validation(ids[:-1], y[:-1], rx[:-1], day[:-1], m.SPLIT_SEED)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    m = api()
    from experiments.pamf_phase3_open_world.ground_open_source import SourceDataset
    from experiments.pamf_phase3_open_world import star_relative_source as native
    counts = dict(L=180, Us=90, OE=60, V=180, novel=45)
    monkeypatch.setattr(m, 'COUNTS', counts); monkeypatch.setattr(m, 'CELL_SIZE', 2)
    source = tmp_path/'source'; source.mkdir()
    idsV, yV, rxV, dayV = validation_rows()
    rng = np.random.default_rng(17); roles = {}
    for role, n in counts.items():
        if role == 'V': ids, y, rx, day = idsV, yV, rxV, dayV
        else:
            ids = np.asarray([role+'-%04d' % i for i in range(n)])
            y = np.arange(n, dtype=np.int64) % 6
            rx = np.asarray(m.RECEIVERS, np.int64)[np.arange(n) % 5]
            day = np.arange(n, dtype=np.int64) % 3 + 1
        meta = dict(ids=ids, rx_i=rx, day_i=day, domain=np.searchsorted(m.RECEIVERS, rx)*3+day-1)
        if role in ('L', 'V'): meta['y'] = y
        iq = rng.normal(size=(n, 2, 256)).astype(np.float32)
        roles[role] = dict(ids=ids, metadata=meta, iq=iq)
        if role in ('L', 'Us', 'V'):
            label = 'L_s' if role == 'L' else 'U_s' if role == 'Us' else 'V'
            np.save(source/(label+'_x.npy'), iq)
            internal = dict(meta, eq_i=np.ones(n, np.int64), sig_i=np.arange(n), base_index=np.arange(n))
            np.savez(source/(label+'_metadata.npz'), **internal)
    contract = dict(classes=native.CLASSES, source_rxs=list(m.RECEIVERS), source_days=[1, 2, 3],
                    counts=dict(L_s=180, U_s=90, V=180), split_seed=392005,
                    num_classes=6, equalized=1, out_len=256, normalize=True,
                    role_ids={k:roles[r]['ids'].tolist() for k, r in [('L_s','L'),('U_s','Us'),('V','V')]})
    save_json(source/'source_contract.json', contract)
    readercfg = dict(base_release=str(tmp_path/'base'), source_output=str(tmp_path/'F0'), model_seed=2026092701,
                     frozen_sources={k:str(tmp_path/k) for k in ('F','AF','BASE','FOLD','G')})
    sc = dict(run_id=m.CACHE_RUN, code_commit='a'*40, reader_config=readercfg, source_cache=str(source),
              source_contract=str(source/'source_contract.json'), raw_mainsig=str(tmp_path/'UNOPENABLE.pkl'),
              mixture_config_path=str(tmp_path/'mixture.json'), episode=dict(native.EPISODE),
              dura_checkpoint=str(tmp_path/'G.pt'), dura_release=str(tmp_path/'G_release'),
              output_root=str(tmp_path/'old_cache'), batch_size=128)
    ref = dict(run_id=m.fresh.RUN_ID, code_commit=m.REFERENCE_CODE, output_root=str(tmp_path/'reference'))
    monkeypatch.setattr(m.fresh, 'validate_config', lambda c:c)
    c = dict(run_id=m.RUN_ID, code_commit='b'*40, reference_config=ref, source_config=sc,
             diverse_data_root=str(tmp_path/'diverse'), oe_data_root=str(tmp_path/'oe'),
             split_seed=m.SPLIT_SEED, output_root=str(tmp_path/'data'), batch_size=128)
    f0 = dict(model='original_F0_E50', parameters=232993, epoch=50, updates=11100, model_seed=2026092701,
              source_output=readercfg['source_output'], source_contract=sc['source_contract'], base_release=readercfg['base_release'])
    dura = dict(deployment_id='dura_v1', checkpoint=sc['dura_checkpoint'], release_root=sc['dura_release'])
    pm = dict(schema=m.fresh.SCHEMA, status=m.fresh.STATUS, run_id=ref['run_id'], code_commit=m.REFERENCE_CODE,
              config=ref, parent_F0_identity=f0, dura_identity=dura, loaded_dura_identity=dura,
              all_three_DURA_conditions_complete=True, all_conditions_through_DURA=True,
              source_models_frozen_before_V=True, all_thresholds_frozen_before_novel_and_Q=True,
              classification_unchanged=True, private_truth_reads=0, target_U_reads=0)
    save_json(Path(ref['output_root'])/'complete.json', pm)
    events = []; counters = dict(G=0,F=0)
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.register_buffer('offset', torch.arange(6, dtype=torch.float32))
        def features(self, x):
            counters['F'] += len(x); return x.mean((1,2))[:, None]
        def classify_features(self, f): return f + self.offset
    reader = Reader().eval().requires_grad_(False)
    g = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    def engine(x): counters['G'] += len(x); return (x*.7+.2,)
    b = dict(reader=reader, dura=(g,engine), reader_identity=f0, dura_identity=dura,
             model=torch.nn.Linear(1,1).eval().requires_grad_(False))
    monkeypatch.setattr(m.fresh, '_backend', lambda *a:b)
    def native_reader(root, role, contract):
        assert (Path(c['output_root'])/'smoke.json').is_file()
        events.append(role)
        return SourceDataset(root, role, contract)
    monkeypatch.setattr(m, 'SourceDataset', native_reader)
    def auxiliary(config, role):
        assert (Path(c['output_root'])/'smoke.json').is_file(); events.append(role)
        r=roles[role];return r['iq'],r['metadata'],dict(count=len(r['ids']),data_status='VALIDATED_ONCE')
    monkeypatch.setattr(m, '_clean_auxiliary', auxiliary)
    def star_role(config, role, marker):
        r=roles[role]
        return dict(ids=r['ids'], metadata=r['metadata'], restored=(r['iq']*-2).astype(np.float32), manifest={})
    monkeypatch.setattr(m.fresh, '_iq_role', star_role)
    scenarios = np.asarray(['practical_high']*counts['V'])
    def old_rows(root, condition, role):
        r=roles['V' if role=='known' else 'novel'];d=dict(ids=r['ids'],default_logits=np.full((len(r['ids']),6),9,np.float32))
        if role=='known':d.update(y=yV,rx_i=rxV,day_i=dayV,scenario=scenarios)
        return d
    monkeypatch.setattr(m.fresh,'_source_rows',old_rows)
    return m,c,roles,b,events,counters


def test_actual_build_blinds_paired_queries_and_keeps_source_roles(fixture, monkeypatch, subtests):
    m,c,roles,b,events,counters=fixture
    load=np.load
    def blind(path,*args,**kwargs):
        if any(k in str(path) for k in ('U_s_y','/truth','known_test_truth','UNOPENABLE')):
            pytest.fail('Producer attempted a private label/raw read')
        return load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',blind)
    done=m.build(c,'cpu');out=Path(c['output_root'])
    assert events == ['L_s','U_s','OE','V','novel']
    assert done['status']==m.STATUS and done['source_only'] and done['private_truth_reads']==done['target_U_reads']==0
    assert counters==dict(G=sum(m.COUNTS.values())+2,F=m.COUNTS['L']+m.COUNTS['V']+m.COUNTS['novel']+2)
    assert done['G_forward_rows']==sum(m.COUNTS.values()) and done['real_F_forward_rows']==405
    for role,n in [('L',180),('Us',90),('OE',60),('CAL',90)]:
        with subtests.test(role=role):
            a=m.load_role(out,role)
            assert len(a['ids'])==n and isinstance(a['iq'],np.memmap) and not a['iq'].flags.writeable
            assert set(a['metadata'])=={'ids','rx_i','day_i','domain'}|({'y'} if role in ('L','CAL') else set())
            assert ('teacher_logits' in a)==(role in ('L','CAL'))
            if role=='L':np.testing.assert_allclose(a['iq'],roles['L']['iq']*.7+.2,rtol=0,atol=0)
            if role in ('L','CAL'):
                expected=a['iq'].mean((1,2),dtype=np.float32)[:,None]+np.arange(6,dtype=np.float32)
                np.testing.assert_allclose(a['teacher_logits'],expected,atol=2e-6,rtol=0)
            m.close(a);assert a['iq']._mmap.closed
    clean=m.load_query(out,'source_clean_DURA');star=m.load_query(out,'source_star_DURA')
    np.testing.assert_array_equal(clean['ids'],star['ids'])
    assert len(clean['ids'])==135 and set(clean['metadata'])=={'ids','rx_i','day_i','domain'}
    assert 'y' not in clean and not any('role' in k or 'unknown' in k for k in clean['metadata'])
    assert np.all(star['baseline_logits']==9)
    assert not np.array_equal(clean['iq'],star['iq'])
    m.close(clean);m.close(star)
    with subtests.test(privacy='Us metadata label leakage'):
        p=out/'roles/Us/metadata.npz';original={k:roles['Us']['metadata'][k] for k in m.META_FIELDS}
        np.savez(p,**dict(original,y=np.zeros(90,np.int64)))
        with pytest.raises(ValueError,match='leak'):m.load_role(out,'Us')
        np.savez(p,**original)
    # The builder writes TEST labels once and never reads them.
    with load(out/'private/known_test_truth.npz',allow_pickle=False) as z:
        assert set(z.files)=={'ids','y','rx_i','day_i','star_scenario'} and len(z['ids'])==90
        cal=m.load_role(out,'CAL');assert not set(z['ids']) & set(cal['ids']);m.close(cal)
    with pytest.raises(FileExistsError):m.build(c,'cpu')


def test_config_rejects_truth_roles_seed_wrong_ancestor_and_overlap(fixture, subtests):
    m,c,*_=fixture
    cases=[dict(c,unknown_truth='forbidden'),dict(c,split_seed=7),dict(c,batch_size=64),
           dict(c,reference_config=dict(c['reference_config'],code_commit='0'*40)),
           dict(c,output_root=c['source_config']['source_cache'])]
    for value in cases:
        with subtests.test(value=list(set(value)-set(c))):
            with pytest.raises(ValueError):m.validate_config(value)
    p=Path(c['reference_config']['output_root'])/'complete.json';d=json.loads(p.read_text());d['parent_F0_identity']['updates']=11101;save_json(p,d)
    with pytest.raises(ValueError):m.build(c,'cpu')


def test_failed_smoke_never_opens_source(fixture,monkeypatch):
    m,c,roles,b,events,counters=fixture
    def fail(*a):raise ValueError('checkpoint smoke failure')
    monkeypatch.setattr(m,'_smoke',fail)
    with pytest.raises(ValueError,match='smoke'):m.build(c,'cpu')
    assert not (Path(c['output_root'])/'complete.json').exists()
    assert events==[] and counters==dict(G=0,F=0)

