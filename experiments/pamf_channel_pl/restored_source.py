"""Frozen source-only raw/channel/DURA pairs; no optimizer or target inputs."""
import argparse
import os
from pathlib import Path
import time

from experiments.cric_dura_comparison.io import read, write
from .plan import SEEDS
from .source_quality import validate_prefix_headers

RUN = '20261025-diagnostic-PAMF-DURA-paired-source-m3-r01'
SCENES = ('practical_low_suburban', 'practical_mid_urban', 'practical_low_urban')
VIEWS = ('raw', 'G_raw') + tuple(v for s in SCENES for v in (s, 'G_'+s))
THRESHOLD = .95


def validate_context(c):
    if set(c) != {'project', 'python', 'frozen_sources', 'cases', 'G'}:
        raise ValueError('Only fixed source handles and G are accepted')
    cases = c['cases']
    if len(cases) != 3 or {x['model_seed'] for x in cases} != set(SEEDS):
        raise ValueError('Exactly three original ordinary-PL seeds required')
    for x in cases:
        if (set(x) != {'row_id', 'protocol', 'model_seed', 'parent_source', 'source_config', 'expected_source_contract'}
                or x['protocol'] != 'F0' or x['row_id'] != 'F0-ordinary_PL-s'+str(x['model_seed'])
                or not x['parent_source'].endswith('/segment20')
                or x['source_config'].get('ablation') != 'F21'
                or x['source_config'].get('num_classes') != 6):
            raise ValueError('Only original F0 ordinary-PL source prefixes required')
    if c['G'].get('kind') != 'original_F0_LU' or c['G'].get('roles') != 'LU':
        raise ValueError('Fixed original F0 G-LU required')
    return sorted(cases, key=lambda x: x['model_seed'])


def validate_e30(case, done, cp, contract, expected):
    c = done['config']; a = cp['args']; parent = case['parent_source']
    if (contract != expected or done.get('status') != 'SOURCE_FROZEN'
            or done.get('epoch') != 10 or done.get('updates') != 2220
            or done.get('initialization') != 'own_previous_segment' or done.get('ancestors') != [parent]
            or done.get('target_access') is not False or done.get('U_truth_read') is not False
            or done.get('head_fit_updates') != 0 or c.get('parent_source') != parent
            or c.get('epoch_offset') != 20 or c.get('step_offset') != 4440
            or c.get('model_seed') != case['model_seed'] or c.get('ablation') != 'F21'
            or any(c.get(k) != case['source_config'].get(k) for k in ('run_id','row_id','num_classes'))
            or cp.get('run_id') != c['run_id'] or cp.get('candidate_id') != c['row_id']
            or cp.get('epoch') != 10 or cp.get('a1_ema_successful_updates') != 2220
            or a.get('baseline_ckpt') != parent+'/final_ssdg.pth' or a.get('teacher_ckpt') != ''
            or a.get('use_unlabeled') is not False or a.get('use_ema_teacher') is not False):
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: own ordinary-PL E20+10 required')


def source_channel(x, meta, scene, contract):
    """Same source practical physics; diagnostic seed fixed by physical record."""
    import numpy as np
    import torch
    from leo_practical.channel import Config, stable_seed, receiver_for_session, normalize_input
    from leo_practical.residual import ResidualChannel
    keys = ('eq_i', 'rx_i', 'day_i', 'sig_i', 'base_index')
    values = {k: meta[k].detach().cpu().tolist() for k in keys}
    if (scene not in SCENES or set(values['rx_i'])-set(contract['source_rxs'])
            or set(values['day_i'])-set(contract['source_days']) or set(values['eq_i']) != {1}):
        raise ValueError('Only declared source physical sessions accepted')
    cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
        mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
    raw = np.asarray(x.detach().cpu().tolist(), np.float64); out = []
    for i in range(len(x)):
        pid = tuple(int(values[k][i]) for k in keys)
        session = 'rx%d:day%d:eq1' % (values['rx_i'][i], values['day_i'][i])
        hw = receiver_for_session(cfg, 20261016423, session)
        signal, _ = normalize_input(raw[i,0]+1j*raw[i,1])
        seed = stable_seed(20261016421, 'PAMF/restored_source/V/fixed', scene, pid)
        signal, info = ResidualChannel(cfg, seed, hw).process(signal)
        if not np.isfinite(signal).all() or info['channel_equalization_applied']:
            raise ValueError('Finite noEQ source view required')
        out.append(np.stack((signal.real, signal.imag)))
    a = np.ascontiguousarray(out, np.float32)
    return torch.frombuffer(bytearray(a.tobytes()), dtype=torch.float32).reshape(a.shape).to(x.device)


def paired_counts(raw, restored, y, mask=None):
    import numpy as np
    m = np.ones(len(y), dtype=bool) if mask is None else mask
    a, b = raw == y, restored == y
    n = int(m.sum()); both = int((m & a & b).sum()); fix = int((m & ~a & b).sum())
    harm = int((m & a & ~b).sum()); wrong = int((m & ~a & ~b).sum())
    return dict(n=n, both_correct=both, corrected=fix, broken=harm, both_wrong=wrong,
        net_correct=fix-harm, raw_accuracy=100.*(both+harm)/n if n else None,
        restored_accuracy=100.*(both+fix)/n if n else None)


def policies(logits, y):
    import numpy as np
    z = logits-logits.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
    confidence = p.max(-1); labels = p.argmax(-1); results = []
    for j, seed in enumerate(SEEDS):
        accepted = confidence[j] >= THRESHOLD
        results.append(dict(policy='single_'+str(seed), accepted=int(accepted.sum()),
            correct=int((accepted & (labels[j] == y)).sum()), seen=len(y)))
    accept = (confidence >= THRESHOLD).all(0) & (labels == labels[:1]).all(0)
    n = int(accept.sum()); correct = int((accept & (labels[0] == y)).sum())
    results.append(dict(policy='unanimous3', accepted=n, correct=correct, seen=len(y),
        precision_percent=100.*correct/n if n else None, coverage_percent=100.*n/len(y),
        retention_vs_mean_single_accepted=n/np.mean([r['accepted'] for r in results]) if any(r['accepted'] for r in results) else 0.,
        feasibility_pass=bool(n and 100.*correct/n >= 99.5 and n >= .9*np.mean([r['accepted'] for r in results]))))
    return results, accept, labels[0]


def snapshot(model):
    return {k:v.detach().clone() for k,v in model.state_dict().items()}


def unchanged(model, before):
    import torch
    return (not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        and all(torch.equal(v, before[k]) for k,v in model.state_dict().items()))


def run(rt):
    if rt['run_id'] != RUN or os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or rt['allowed_gpus'] != [0]:
        raise ValueError('Fixed source diagnostic GPU0 only')
    from experiments.pamf_repl_completion.runner import activate
    activate(rt)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload, load_G
    from experiments.cric_dura_comparison.external_dura import fp32_copy
    from experiments.pamf_stage2_inference.engine import InferenceEngine
    torch.set_num_threads(2)
    c = read(rt['context']); cases = validate_context(c); root = Path(rt['run_root'])
    root.mkdir(parents=True, exist_ok=False); start = time.perf_counter()
    write(root/'pipeline_state.json', dict(status='SOURCE_LOADING',pid=os.getpid(),code_commit=rt['code_commit']))
    models=[]; configs=[]; contract=None
    for case in cases:
        folder=Path(case['parent_source']); d20=read(folder/'complete.json')
        expected=read(case['expected_source_contract']); actual=read(d20['config']['source_contract'])
        cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
        validate_prefix_headers(case,d20,cp,actual,expected);del cp
        folder=folder.parent/'segment30';done=read(folder/'complete.json');cfg=done['config']
        cp=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False)
        actual=read(cfg['source_contract']);validate_e30(case,done,cp,actual,expected)
        if (contract is not None and actual != contract) or cfg['source_rxs'] != actual['source_rxs'] or cfg['source_days'] != actual['source_days']:
            raise ValueError('Identical source role/receiver/day contracts required')
        contract=actual;models.append(reader_from_payload(cfg,cp,'cuda:0'));configs.append(cfg);del cp
    if contract['counts'] != dict(L_s=6300,U_s=56700,V=27000) or any(cfg['source_cache'] != configs[0]['source_cache'] for cfg in configs):
        raise ValueError('Original F0 source pool required')
    g=load_G(c['G'],contract,'cuda:0');engine=InferenceEngine(g,reuse=True,graph=True,native=False)
    labeled=SourceDataset(configs[0]['source_cache'],'L_s',contract);x=labeled.x[:2].to('cuda:0')
    with torch.inference_mode():
        eager=g(x);restored=engine(x)[0].clone();repeat=engine(x)[0].clone()
        if not torch.allclose(eager,restored,atol=1e-5,rtol=1e-5) or not torch.equal(restored,repeat):
            raise ValueError('Fixed deployed DURA route/repeatability smoke failed')
        for model in models:
            z=model.features(restored);q=model.classify_features(z)
            if z.shape != (2,160) or q.shape != (2,6) or not bool(torch.isfinite(q).all()) or not torch.allclose(q,model(restored),atol=1e-5,rtol=1e-5):
                raise ValueError('Real source default-head feature smoke failed')
    del labeled,x,eager,restored,repeat
    before=[snapshot(m) for m in models];gbefore=snapshot(g)
    write(root/'smoke.json',dict(status='PASS',role='L_s',physical_packets=2,frozen_F=3,frozen_G=1,
        G_eager_graph_close=True,G_repeat_identical=True,default_head=True,optimizer_updates=0,target_reads=0,U_truth_read=False))
    data=SourceDataset(configs[0]['source_cache'],'V',contract);n=len(data)
    loader=torch.utils.data.DataLoader(data,batch_size=256,shuffle=False,num_workers=0)
    logits=np.empty((3,len(VIEWS),n,6),np.float32);cos=np.empty((3,len(VIEWS),n),np.float32)
    power=np.empty((len(VIEWS),n),np.float32);rxs=sorted(contract['source_rxs']);geometry={}
    for j in range(3):
        for v in range(len(VIEWS)):
            geometry[j,v]=dict(sums=np.zeros((6,len(rxs),160),np.float64),counts=np.zeros((6,len(rxs)),np.int64))
    offset=0
    for i,batch in enumerate(loader):
        x=batch[0].to('cuda:0');count=len(x);y=batch[1].tolist();rx=batch[3]['rx_i'].tolist()
        with torch.inference_mode():
            views=[x,engine(x)[0].clone()]
            for scene in SCENES:
                h=source_channel(x,batch[3],scene,contract);views.extend((h,engine(h)[0].clone()))
            for v,view in enumerate(views):
                if not bool(torch.isfinite(view).all()):raise ValueError('Nonfinite restored source IQ')
                power[v,offset:offset+count]=fp32_copy(view.square().sum(1).mean(1))
            for j,model in enumerate(models):
                reference=None
                for v,view in enumerate(views):
                    feature=model.features(view);q=model.classify_features(feature);z=F.normalize(feature,dim=1)
                    if q.shape != (count,6) or not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(q).all()):
                        raise ValueError('Finite default-head logits/features required')
                    if reference is None:reference=z
                    logits[j,v,offset:offset+count]=fp32_copy(q)
                    cos[j,v,offset:offset+count]=fp32_copy(1-(z*reference).sum(1))
                    features=fp32_copy(z);summary=geometry[j,v]
                    for k,(label,r) in enumerate(zip(y,rx)):
                        rindex=rxs.index(r);summary['sums'][label,rindex]+=features[k];summary['counts'][label,rindex]+=1
        offset+=count
        if i%5 == 0 or offset==n:
            write(root/'progress.json',dict(status='SOURCE_PAIRED_INFERENCE',source_V_seen=offset,total=n,
                F_packet_forwards=3*len(VIEWS)*offset,G_packet_forwards=4*offset,
                seconds=time.perf_counter()-start,model_updates=0,target_reads=0),exclusive=False)
            print('SOURCE_PAIRED',offset,n,flush=True)
    if offset != 27000 or not unchanged(g,gbefore) or any(not unchanged(m,b) for m,b in zip(models,before)):
        raise ValueError('Incomplete source V or model parameter/buffer mutation')
    truth=data.metadata['y'];pol,accepted,pseudo=policies(logits[:,0],truth);rows=[]
    for j,seed in enumerate(SEEDS):
        for v,name in enumerate(VIEWS):
            pred=logits[j,v].argmax(-1);q=logits[j,v];margin=q[np.arange(n),truth]-np.max(np.where(np.eye(6,dtype=bool)[truth],-np.inf,q),axis=1)
            dimensions=[('overall','ALL',np.ones(n,bool)),('raw_consensus','accepted',accepted)]
            for dim,values in [('TX',truth),('RX',data.metadata['rx_i']),('day',data.metadata['day_i'])]:
                dimensions.extend((dim,str(value),values==value) for value in np.unique(values))
            for dim,stratum,mask in dimensions:
                rows.append(dict(model_seed=seed,view=name,dimension=dim,stratum=stratum,
                    **paired_counts(logits[j,0].argmax(-1),pred,truth,mask),
                    mean_true_class_margin=float(margin[mask].mean()) if mask.any() else None,
                    mean_paired_cosine_distance=float(cos[j,v,mask].mean()) if mask.any() else None,
                    mean_IQ_power=float(power[v,mask].mean()) if mask.any() else None))
    geom=[]
    for (j,v),s in geometry.items():
        centroids=s['sums']/np.maximum(s['counts'][...,None],1);norm=np.linalg.norm(centroids,axis=-1)
        unit=centroids/np.maximum(norm[...,None],1e-12);within=[];between=[]
        for a in range(6):
            for r in range(len(rxs)):
                for t in range(r+1,len(rxs)):within.append(1-float(unit[a,r]@unit[a,t]))
                for b in range(a+1,6):between.append(1-float(unit[a,r]@unit[b,r]))
        geom.append(dict(model_seed=SEEDS[j],view=VIEWS[v],mean_class_RX_resultant_length=float(norm.mean()),
            same_TX_cross_RX_centroid_distance=float(np.mean(within)),different_TX_same_RX_centroid_distance=float(np.mean(between)),
            fitted_probe_updates=0))
    np.savez(root/'source_V_predictions.npz',logits=logits,y=truth,ids=data.metadata['ids'],
        rx=data.metadata['rx_i'],day=data.metadata['day_i'],views=np.asarray(VIEWS),seeds=np.asarray(SEEDS),
        paired_cosine_distance=cos,IQ_power=power,raw_consensus_accepted=accepted,raw_consensus_label=pseudo)
    result=dict(status='SOURCE_PAIRED_DIAGNOSTIC_COMPLETE',code_commit=rt['code_commit'],source_V_seen=n,
        model_seeds=list(SEEDS),parent_epoch=30,G_roles='LU',views=list(VIEWS),policies=pol,results=rows,geometry=geom,
        F_packet_forwards=3*len(VIEWS)*n,G_packet_forwards=4*n,smoke_cost_separate=True,
        source_V_training_reads=0,U_IQ_reads=0,U_truth_read=False,target_reads=0,
        optimizer_updates=0,F_updates=0,G_updates=0,head_fit_updates=0,parameters_and_buffers_unchanged=True,
        source_channel_seed=20261016421,source_hardware_seed=20261016423,
        source_namespace='PAMF/restored_source/V/fixed',source_IQ_secondary_normalization=False,
        seconds=time.perf_counter()-start,scope='Source-V diagnosis only; not new SSL training or target performance.')
    write(root/'complete.json',result)
    write(root/'pipeline_state.json',dict(status=result['status'],pid=os.getpid(),code_commit=rt['code_commit']))
    print('SOURCE_PAIRED_COMPLETE',result['seconds'],flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['dispatch']);p.add_argument('--config',required=True)
    a=p.parse_args();run(read(a.config))
