"""Focused NumPy/metadata negatives for the raw-IQ frozen source reader."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np


def reader():
    path = Path(__file__).with_name('cvs_raw_iq_adapter_source_fit_20261019.py')
    spec = importlib.util.spec_from_file_location('_rawfit_checks_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(r):
    selected, index = [], []
    for tx in range(6):
        for rx in (1, 3, 4, 6, 8):
            for day in (1, 2, 3):
                for sig in range(2):
                    i = len(index)
                    identity = format(i, '024x')
                    index.append(SimpleNamespace(tx_i=tx, eq_i=1, rx_i=rx, day_i=day,
                                                 sig_i=sig, physical_id=identity))
                    selected.append(dict(TX=tx, RX=rx, day=day, pid=[1, rx, day, sig, i], physical_id=identity))
    base = SimpleNamespace(index=index, tx_list=[str(i) for i in range(6)], transform=None, normalize=True, center=False)
    origins = [dict(local_step=2220, cumulative_step=6660, local_epoch=10, epoch=30, slot=i % 128,
                    pid=s['pid'], physical_id=s['physical_id'], namespace='same-train-namespace',
                    scene='practical_low_urban', step_scene_counts_verified=True) for i, s in enumerate(selected)]
    records = []
    for vi, view in enumerate(r.VIEWS[1:], 1):
        for i, s in enumerate(selected):
            records.append(dict(view=view, panel_index=i, physical_id=s['physical_id'], sample_id=s['pid'],
                TX=s['TX'], RX=s['RX'], day=s['day'], scene='practical_low_urban', namespace='same-draw'+str(vi),
                session_id='same-session', seed=i+vi))
    channel = dict(views=list(r.VIEWS), shape=[180, 6, 2, 256], training_scenes=['practical_low_urban'],
        channel_seeds=dict(train=1, receiver=2), fresh_namespaces=['draw0', 'draw1'], sessions={'x':{'hardware':1}},
        native_normalizations=900, extra_RMS=0, output_normalization=False, generated_packets=900,
        observed_last_packets=180, fresh_packets=720, raw_packets=180,
        channel_audit=dict(configurations={'scene':{'fs_hz':25000000.,'equalization_enabled':False}}), records=records)
    panel = dict(selected=copy.deepcopy(selected), origins=copy.deepcopy(origins), channel=copy.deepcopy(channel),
                 views=list(r.VIEWS), source_raw_reads=180, target_access=False)
    complete = dict(status='SOURCE_ONLY_ZERO_UPDATE_COMPLETE', target_access=False)
    labels = np.asarray([s['TX'] for s in selected], dtype=np.int64)
    logits = np.zeros((180, 6, 6)); logits[np.arange(180), :, labels] = 1.
    arrays = dict(TC_logits=logits.copy(), TS_logits=logits.copy(), TC_joint=np.zeros((180, 6, 160)),
        TX=labels, ids=np.asarray([s['physical_id'] for s in selected]),
        RX=np.asarray([s['RX'] for s in selected]), day=np.asarray([s['day'] for s in selected]))
    readouts = {n:dict(model=n, origin='fixed E20' if n=='TC' else 'fixed E200', views=list(r.VIEWS),
        n_per_view=180, NN_packets=1080, registered_state_unchanged=True, target_access=False,
        optimizer_updates=0, correct=[180]*6) for n in ('TC','TS')}
    return selected, base, origins, channel, panel, complete, arrays, readouts


def run():
    r = reader()
    selected, base, origins, channel, panel, complete, arrays, readouts = fixture(r)
    roles = dict(L_s=[s['physical_id'] for s in selected])
    opaque = lambda item: item.physical_id
    r.checked_panel(selected, base, roles, opaque, base.tx_list)
    value = r.checked_reuse(panel, complete, readouts, arrays, selected, origins, channel)
    assert value['new_NN_packets'] == 0 and r.MAX_NN == 4*180*6+1
    negatives = []
    def reject(name, call):
        try:
            call()
        except (ValueError, KeyError, IndexError):
            negatives.append(name)
        else:
            raise AssertionError('Expected rejection: '+name)
    for field, bad in [('normalize',False),('center',True),('transform',object())]:
        b = copy.copy(base); setattr(b,field,bad)
        reject('raw preprocessing '+field,lambda b=b:r.checked_panel(selected,b,roles,opaque,base.tx_list))
    s=copy.deepcopy(selected);s[0]['physical_id']='not-L'
    reject('unauthorized physical L',lambda:r.checked_panel(s,base,roles,opaque,base.tx_list))
    s=copy.deepcopy(selected);s[0]['pid'][4]=999999
    reject('physical index before IQ',lambda:r.checked_panel(s,base,roles,opaque,base.tx_list))
    def reuse(p=panel,c=complete,rd=readouts,a=arrays,o=origins,ch=channel):
        return r.checked_reuse(p,c,rd,a,selected,o,ch)
    for key,bad in [('namespace','other'),('slot',127),('scene','practical_mid_urban'),('cumulative_step',6659)]:
        changed=copy.deepcopy(origins);changed[0][key]=bad
        reject('last origin '+key,lambda changed=changed:reuse(o=changed))
    for key,bad in [('fresh_namespaces',['different','draw1']),('channel_seeds',dict(train=2,receiver=2)),
                    ('output_normalization',True),('sessions',{'x':{'hardware':2}})]:
        changed=copy.deepcopy(channel);changed[key]=bad
        reject('channel '+key,lambda changed=changed:reuse(ch=changed))
    changed=copy.deepcopy(channel);changed['records'][0]['seed']+=1
    reject('individual channel seed',lambda:reuse(ch=changed))
    changed=copy.deepcopy(channel);changed['channel_audit']['configurations']['scene']['equalization_enabled']=True
    reject('physical configuration',lambda:reuse(ch=changed))
    changed=copy.deepcopy(arrays);changed['ids']=changed['ids'][::-1]
    reject('prediction physical order',lambda:reuse(a=changed))
    changed=copy.deepcopy(arrays);changed['TC_logits']=changed['TC_logits'][:179]
    reject('cached logit shape',lambda:reuse(a=changed))
    changed=copy.deepcopy(arrays);changed['TS_logits'][0,0,0]=np.nan
    reject('cached logit finite',lambda:reuse(a=changed))
    changed=copy.deepcopy(arrays);changed['TC_joint']=changed['TC_joint'][:,:,:159]
    reject('joint160 shape',lambda:reuse(a=changed))
    changed=copy.deepcopy(readouts);changed['TS']['registered_state_unchanged']=False
    reject('teacher persistent state',lambda:reuse(rd=changed))
    changed=copy.deepcopy(readouts);changed['TC']['correct'][0]-=1
    reject('teacher readout vs logits',lambda:reuse(rd=changed))
    changed=dict(complete,target_access=True)
    reject('teacher cache target contact',lambda:reuse(c=changed))
    # Full-length source-sequence interface, only synthetic metadata files.
    with tempfile.TemporaryDirectory() as root:
        paths=[Path(root)/(str(i)+'.gz') for i in range(4)]
        rows=[dict(role='L' if i%2==0 else 'U',batch=i//2+1,ids=[[1,1,1,0,0]]) for i in range(4440)]
        def save(path,values):
            with gzip.open(path,'wt',encoding='utf8') as f:
                for row in values:f.write(json.dumps(row)+'\n')
        for path in paths:save(path,rows)
        assert r.same_sequences(paths)['matching_logged_batches_per_row']==4440
        save(paths[-1],rows[:-1])
        reject('truncated sampler stream',lambda:r.same_sequences(paths))
        changed=copy.deepcopy(rows);changed[0]['ids'][0][-1]=1;save(paths[-1],changed)
        reject('different actual sampler',lambda:r.same_sequences(paths))
    return dict(status='PASS',focused_negative_cases=negatives,negative_count=len(negatives),
        frozen_cache_reuse=True,NN_budget=4321,teacher_new_NN=0,channel_packets=900,
        synthetic_only=True,CP_reads=0,actual_IQ_reads=0,NN=0,remote=0,updates=0,target_access=False)


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,allow_nan=False))
