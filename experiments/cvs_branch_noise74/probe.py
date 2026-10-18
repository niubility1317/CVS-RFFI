"""Observe two frozen models; swap only whole contributions, never logits."""
import gzip
import json
import os
import sys
import time
import traceback
import numpy as np
import torch
from . import design as d
from .capture import observe, interventions

def tensor_iq(array):
    a = np.ascontiguousarray(array, dtype=np.float32)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape)

def array(tensor):
    return np.asarray(tensor.detach().cpu().tolist(), dtype=np.float32)

def clean_pack(p, begin, end):
    # Packs retain the exact model/version signature captured by observe.
    q = dict(p)
    q['logits'], q['joint'] = p['logits'][begin:end], p['joint'][begin:end]
    q['contributions'] = {k:v[begin:end] for k,v in p['contributions'].items()}
    return q

def combine(packs):
    p = dict(packs[0])
    for key in ('logits', 'joint'):
        p[key] = torch.cat([q[key] for q in packs])
    p['contributions'] = {k:torch.cat([q['contributions'][k] for q in packs]) for k in d.BRANCHES}
    return p

def panel():
    from experiments.cvs_crossview_supcon41.quality import panel as old_panel
    data, roles = old_panel(torch)
    done = d.read(d.PANEL/'observations/completion.json')
    engine = d.validate_physics(d.read(d.PANEL/'observations/actual_engine.json'))
    if (done['conditions'][:2] != ['full', 'no_noise/fixed'] or done['draws'] != 2
            or done['scenes'] != ['practical_low_suburban', *d.SCENES]
            or done['target_access'] or done['training_updates']):
        raise ValueError('Original fixed-receiver source observations required')
    iq = np.load(d.PANEL/'observations/iq.npy', mmap_mode='r', allow_pickle=False)
    if iq.shape != (2,3,11,d.N,2,256) or iq.dtype != np.float32:
        raise ValueError('Existing observation array differs')
    records = []
    with gzip.open(d.PANEL/'observations/physical_records.jsonl.gz', 'rt', encoding='utf8') as f:
        for line in f:
            r = json.loads(line)
            if r['scene'] not in d.SCENES:
                continue
            i = r['index']
            if (r['draw'] not in (0,1) or not 0 <= i < d.N
                    or r['physical_id'] != data['ids'][i]
                    or (r['TX'],r['RX'],r['day']) != (int(data['y'][i]),int(data['rx'][i]),int(data['day'][i]))):
                raise ValueError('Physical ID and label alignment differs')
            a,b = r['conditions']['full'],r['conditions']['no_noise/fixed']
            if b['effect'] != 'fixed_receiver_state_and_compensation':
                raise ValueError('Must use controlled noise effect, not recomputed receiver state')
            for k in ('state','receiver_lock_status','residual_scale','output_frequency_hz',
                    'estimated_frequency_hz','phase_tracking_applied','channel_equalization_applied',
                    'agc_gain_db','residual_kernel_energy','numerical_delay','boundary'):
                if a[k] != b[k]:
                    raise ValueError('Fixed-state control changed receiver or propagation: '+k)
            # Analysis needs full-state strata, not all eleven old controls.
            records.append({**{k:v for k,v in r.items() if k!='conditions'},
                'conditions':{'full':a},'fixed_receiver_control_verified':True})
    keys = {(r['draw'],r['scene'],r['index']) for r in records}
    if len(records) != 7200 or len(keys) != 7200:
        raise ValueError('Complete balanced V panel/draw matrix required')
    d.write(d.BASE/'panel_metadata.json', dict(full_records=records, data_roles=roles,
        physics=engine, cache_reused=True, new_physical_simulations=0,
        diagnostic_fc_hz=2450000000., parent_training_fc_hz=2462000000.,
        target_access=False, U_truth_read=False,scene_order=list(d.SCENES),
        branch_order=list(d.BRANCHES),operation_order=list(d.OPERATIONS),
        axis_names=dict(full=['draw','scene','packet','class'],
            interventions=['draw','scene','operation','branch','packet','class'])))
    return data, iq

def model(row):
    from experiments.cvs_noise_shape_factorial43 import model as backend, design as old
    c = old.config(next(r for r in old.rows() if r['arm']==d.ARMS[row]),30)
    m = backend.own_model(c, 'cpu')
    h = m.id_backbone.encoder.core.id_backbone.cls_head
    if (sum(p.numel() for p in m.parameters()) != 900722 or len(m.state_dict()) != 206
            or h.weight.shape != (6,160) or h.scale != 30.
            or type(m.id_backbone.encoder.response).__name__ != 'NoExplicitCFO'
            or tuple(m.id_backbone.encoder.sequence_pool.modes) != ('SIG_TIME',)*3):
        raise ValueError('Original F/SIG_TIME_X3/NoExplicitCFO/defaultcos30 model required')
    return m, c

def run():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only source diagnosis required')
    torch.set_num_threads(2)
    tic = time.perf_counter()
    d.write(d.BASE/'resolved_config.json', d.validate_config(d.config()))
    d.write(d.BASE/'runtime.json',dict(pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,
        CUDA_VISIBLE_DEVICES='',threads=2))
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_noise_shape_factorial43 import design as old
    from experiments.cvs_phase1_overlay.model import native_modules
    native = native_modules()
    actual = dict(SSDG=native.__file__)
    for name in ('leo_practical.channel','leo_practical.residual'):
        mod = __import__(name,fromlist=['__file__']);actual[name]=mod.__file__
        expected = d.PROJECT/'releases/cvs_SIGTIME_evidence_responsibility_train20_20261015_r02/experiments/adv3b02_xuc/code'/Path(*name.split('.')).with_suffix('.py')
        if Path(mod.__file__).resolve() != expected.resolve():
            raise ValueError('Actual leo_practical is not pinned adv3b02_xuc runtime')
    d.write(d.BASE/'actual_modules.json',actual)
    print('ACTUAL_MODULES '+json.dumps(actual),flush=True)
    counts = dict(native_model_examples=0,additional_head_only_examples=0)
    smokes = []
    with numerical_context(old.FULL_FP32_POLICY):
        models = {}
        for row in d.ROWS:
            m,c = model(row);models[row]=(m,c)
            before = {k:v.clone() for k,v in m.state_dict().items()}
            x = torch.randn(2,2,256,generator=torch.Generator().manual_seed(1974))
            p = observe(m,x)
            z = interventions(m,p,p,p)
            if (m.id_backbone.encoder._full_eval_packets != 2 or m.id_backbone.encoder._full_train_packets
                    or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items())
                    or any(v.grad is not None for v in m.parameters())):
                raise ValueError('Real checkpoint no-query smoke mutated state or counts')
            counts['native_model_examples'] += 2
            counts['additional_head_only_examples'] += p['extra_head_examples']+z['extra_head_examples']
            smokes.append(dict(row=row,status='PASS',native_model_examples=2,
                additional_head_only_examples=26,source_IQ=0,query_IQ=0,
                checkpoint=c['output_root']+'/final_ssdg.pth',model_state_unchanged=True,
                joint_error=p['joint_max_abs_error'],logit_error=p['logit_max_abs_error']))
        d.write(d.BASE/'checkpoint_noquery_smoke.json',dict(status='PASS',rows=smokes,
            full_model_examples=4,additional_head_only_examples=52,training_updates=0,
            head_fit_updates=0,target_access=False))
        print('SMOKE PASS 2 frozen checkpoints/4model examples/0query; continue',flush=True)
        data,iq = panel()
        for row,(m,c) in models.items():
            out = d.BASE/row;out.mkdir(exist_ok=False)
            begin_time = time.perf_counter()
            before = {k:v.clone() for k,v in m.state_dict().items()}
            packs = []
            for i in range(0,d.N,d.BATCH):
                packs.append(observe(m,data['x'][i:i+d.BATCH]))
            cp = combine(packs)
            clean_zero = np.empty((4,d.N,6),np.float32)
            for i in range(0,d.N,d.BATCH):
                p = clean_pack(cp,i,i+d.BATCH)
                # Interventions returns all12 but clean controls need only zero4.
                # Use original head directly for four native-order recompositions.
                head = m.id_backbone.encoder.core.id_backbone.cls_head
                with torch.no_grad():
                    for bi,b in enumerate(d.BRANCHES):
                        parts = dict(p['contributions']);parts[b]=torch.zeros_like(parts[b])
                        joint = parts['TF']+parts['PA'];joint=joint+parts['REF'];joint=joint+parts['SIG']
                        clean_zero[bi,i:i+d.BATCH]=array(head.classify(joint))
            counts['native_model_examples'] += d.N
            counts['additional_head_only_examples'] += 5*d.N
            full = np.empty((2,2,d.N,6),np.float32)
            nonoise = np.empty_like(full)
            interventions_out = np.empty((2,2,3,4,d.N,6),np.float32)
            for draw in range(2):
                for si,scene in enumerate(d.SCENES):
                    for i in range(0,d.N,d.BATCH):
                        p = clean_pack(cp,i,i+d.BATCH)
                        f = observe(m,tensor_iq(iq[draw,si+1,0,i:i+d.BATCH]))
                        n = observe(m,tensor_iq(iq[draw,si+1,1,i:i+d.BATCH]))
                        z = interventions(m,f,n,p)
                        full[draw,si,i:i+d.BATCH]=array(f['logits'])
                        nonoise[draw,si,i:i+d.BATCH]=array(n['logits'])
                        for oi,op in enumerate(d.OPERATIONS):
                            for bi,b in enumerate(d.BRANCHES):
                                interventions_out[draw,si,oi,bi,i:i+d.BATCH]=array(z['logits'][op+'_'+b])
                        counts['native_model_examples'] += f['native_model_examples']+n['native_model_examples']
                        counts['additional_head_only_examples'] += f['extra_head_examples']+n['extra_head_examples']+z['extra_head_examples']
                    d.write(d.BASE/'progress.json',dict(row=row,draw=draw,scene=scene,
                        **counts,training_updates=0,target_access=False,elapsed_seconds=time.perf_counter()-tic))
                    print(row,draw,scene,'complete',counts,flush=True)
            if (m.id_backbone.encoder._full_eval_packets != 16202 or m.id_backbone.encoder._full_train_packets
                    or any(not torch.equal(v,m.state_dict()[k]) for k,v in before.items())
                    or any(v.grad is not None for v in m.parameters())):
                raise ValueError('Frozen model changed or actual full-model readout count differs')
            if not all(np.isfinite(a).all() for a in (clean_zero,full,nonoise,interventions_out)):
                raise ValueError('Every actual default-head readout must be finite')
            np.savez_compressed(out/'predictions.npz',ids=np.asarray(data['ids']),
                TX=data['y'].tolist(),RX=data['rx'].tolist(),day=data['day'].tolist(),
                clean=array(cp['logits']),clean_zero=clean_zero,full=full,
                no_noise_fixed=nonoise,interventions=interventions_out)
            ref=m.id_backbone.encoder;head=ref.core.id_backbone.cls_head
            d.write(out/'complete.json',dict(status='SOURCE_CONTRIBUTIONS_COMPLETE',row=row,
                source_config=c,NN=16202,additional_head_only_examples=109826,
                wall_seconds=time.perf_counter()-begin_time,state_unchanged=True,
                training_updates=0,head_fit_updates=0,target_access=False,
                static_gains=dict(PA=head.gain.tanh().tolist(),REF=ref.response_gain.tanh().tolist(),
                    SIG=(.2*ref.sequence_pool.gain_raw.tanh()).tolist()),
                model_seed=2026092701,channel_draws=2,model_parameters=900722,
                clean_from_same_physical_ids=True,normalization='Original native perpacket RMS; no extra post-channel RMS'))
    if counts != dict(native_model_examples=d.NN,additional_head_only_examples=d.EXTRA_HEAD):
        raise ValueError('Actual full-model and additional-head counts differ from plan')
    d.write(d.BASE/'complete.json',dict(status='SOURCE_OBSERVATIONS_COMPLETE',**counts,
        training_updates=0,head_fit_updates=0,new_physical_simulations=0,
        model_seed_count=1,channel_draws=2,rows=list(d.ROWS),target_access=False,source_only=True,
        formal_tests_this_diagnostic=0,goal_complete=False,wall_seconds=time.perf_counter()-tic,
        code_commit=(d.ROOT/'release_commit.txt').read_text().strip()))
    print('SOURCE_OBSERVATIONS_COMPLETE '+json.dumps(counts),flush=True)

def main():
    owned = False
    try:
        d.BASE.mkdir(exist_ok=False);owned=True
        run()
    except BaseException as error:
        if owned:
            d.write(d.BASE/'failure.json',dict(status='FAILED',error=repr(error),
                traceback=traceback.format_exc(),preserve_outputs=True,no_automatic_retry=True))
        raise
