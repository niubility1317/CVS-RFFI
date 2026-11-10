"""Frozen source-only L branch gradients and V directional checks; no model update."""
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import time
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F

RUN = '20261110-diagnostic-pamf-lbranch-guard-source-s2026092701-r01'
ROOT = Path('/home/szu2070436088/2510044040/CV-SincNet')
PARENT = ROOT / 'runs/20261027-phase1-pamf-repl-shuffled-views-m16-r02'
ROW = 'main_split-repl-shuffled_time-3view-s2026092701'
RELEASE = ROOT / 'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
RXS = (1, 3, 4, 6, 8)
BRANCHES = ('time', 'frequency', 'pa', 'path', 'reference')
ORDINALS = tuple(range(0, 60, 5))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


@contextmanager
def capture(core, sequence, reference):
    modules = {'time': core.t_proj, 'frequency_main': core.f_proj,
               'frequency_stats': core.freq_stats_proj, 'pa_main': core.pa_proj,
               'pa_stats': core.pa_stats_proj, 'path': sequence, 'reference': reference}
    values, packet, hooks = {}, {}, []
    def recorder(name):
        def record(_module, _inputs, output):
            if name in values or not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError('Expected one actual branch vector')
            values[name] = output
        return record
    try:
        for name, module in modules.items():
            if module is None: raise ValueError('Missing actual branch')
            hooks.append(module.register_forward_hook(recorder(name)))
        yield packet
        if set(values) != set(modules): raise ValueError('Actual branch bypassed')
        packet.update(time=values['time'], frequency=values['frequency_main'] + values['frequency_stats'],
                      pa=values['pa_main'] + .25 * values['pa_stats'], path=values['path'],
                      reference=values['reference'])
    finally:
        for hook in hooks: hook.remove()


def relation_loss(x, labels, rx, days):
    """Same-day cross-RX same-TX positives; same-RX different-TX negatives."""
    n = len(labels)
    if x.ndim != 2 or len(x) != n or rx.shape != labels.shape or days.shape != labels.shape:
        raise ValueError('Invalid branch relation inputs')
    z = F.normalize(x, dim=1, eps=1e-8)
    same_y = labels[:, None] == labels[None, :]
    same_rx = rx[:, None] == rx[None, :]
    same_day = days[:, None] == days[None, :]
    positive = same_y & ~same_rx & same_day
    negative = ~same_y & same_rx & same_day
    valid = positive.any(1) & negative.any(1)
    if not bool(valid.all()): raise ValueError('Every fixed L anchor must have positive and negative')
    # Equivalent to beta=0 normalized squared-distance contrast, apart from
    # the common constant cancelled by logsumexp.
    score = z @ z.T / .2
    denominator = torch.logsumexp(score.masked_fill(~(positive | negative), -torch.inf), 1)
    positive_mean = score.masked_fill(~positive, 0.).sum(1) / positive.sum(1)
    return .01 * (denominator - positive_mean).mean()


def grads(loss, params, retain=False):
    values = torch.autograd.grad(loss, params, retain_graph=retain, allow_unused=True)
    return [None if v is None else v.detach().double().cpu() for v in values]


def dot(a, b, ids):
    return sum(float((a[i] * b[i]).sum()) for i in ids if a[i] is not None and b[i] is not None)


def norm(a, ids):
    return dot(a, a, ids) ** .5


def protected(aux, base):
    ids = [i for i, g in enumerate(aux) if g is not None]
    if not ids: raise ValueError('Branch loss did not reach a parameter')
    d, b2 = dot(aux, base, ids), dot(base, base, ids)
    factor = min(d / b2, 0.) if b2 > 0 else 0.
    # Restrict protection to parameters actually reached by this branch loss;
    # do not create updates for unrelated paths or the classifier.
    result = [None if g is None else (g - factor * base[i] if base[i] is not None else g)
              for i, g in enumerate(aux)]
    return ids, result, d < 0, factor


def metrics(aux, base, vgrads):
    ids, guard, active, factor = protected(aux, base)
    na, nb, ng = norm(aux, ids), norm(base, ids), norm(guard, ids)
    db = dot(aux, base, ids)
    answer = dict(support_parameters=sum(aux[i].numel() for i in ids),
        aux_norm=na, base_norm_on_aux_support=nb,
        aux_to_base_norm=na/nb if nb else None,
        ce_cosine=db/(na*nb) if na and nb else None,
        ce_dot=db, guard_active=active, guard_projection_factor=factor,
        guarded_ce_dot=dot(guard,base,ids), guarded_norm=ng,
        guard_retained_norm_fraction=ng/na if na else None, validation={})
    for rx, v in vgrads.items():
        nv = norm(v,ids)
        raw, guarded = dot(aux,v,ids), dot(guard,v,ids)
        answer['validation'][str(rx)] = dict(base_norm=nv, raw_dot=raw, guarded_dot=guarded,
            raw_cosine=raw/(na*nv) if na and nv else None,
            guarded_cosine=guarded/(ng*nv) if ng and nv else None)
    return answer


def select_cells(data):
    m = data.metadata
    labels, rx, days = (np.asarray(m[k]) for k in ('y','rx_i','day_i'))
    cells = []
    for y in range(6):
        for r in RXS:
            for d in (1,2,3):
                ids = np.flatnonzero((labels==y) & (rx==r) & (days==d))
                if len(ids) <= max(ORDINALS): raise ValueError('Incomplete source cell')
                cells.append(ids)
    return [np.asarray([ids[ordinal] for ids in cells], dtype=np.int64) for ordinal in ORDINALS]


def batch(data, indices):
    x = data.x.index_select(0, torch.as_tensor(indices, dtype=torch.long)).cpu()
    m = data.metadata
    y, rx, day = (torch.as_tensor(np.asarray(m[k])[indices].copy(), dtype=torch.long)
                  for k in ('y','rx_i','day_i'))
    if x.shape != (90,2,256) or len(y.unique()) != 6: raise ValueError('Bad balanced batch')
    return x, y, rx, day


def main():
    out = ROOT/'runs'/RUN
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = read(PARENT/'configs'/('train__'+ROW+'.json'))
    if cfg['model_seed']!=2026092701 or cfg['source_rxs']!=list(RXS) or cfg['source_days']!=[1,2,3]:
        raise ValueError('Unexpected source contract')
    sys.path.insert(0,str(RELEASE))
    from experiments.pamf_repl_completion.runner import activate
    activate(cfg)
    from experiments.pamf_repl_view_shuffle.source import load_reader
    from experiments.cric_dura_comparison.f_data import SourceDataset
    contract = read(cfg['source_contract'])
    reader,_ = load_reader(cfg['output_root'],contract,'cpu')
    reader.eval().requires_grad_(True)
    # The frozen prediction reader normally computes filters under no_grad.
    # Restore the original differentiable synthesis for this in-memory probe.
    from experiments.cvs_rdrop_sinc_units.coordinate import CoordinateSinc
    sinc = reader.core.id_backbone.sinc
    if not isinstance(sinc, CoordinateSinc): raise ValueError('Unexpected Sinc implementation')
    sinc._filters = MethodType(CoordinateSinc._filters, sinc)
    sinc._coordinate_cache_key = sinc._filter_cache_key = sinc._filter_cache = None
    params = list(reader.parameters())
    before = [p.detach().clone() for p in params]
    if any(p.grad is not None for p in params): raise ValueError('Preexisting parameter gradients')
    # Real checkpoint smoke; no input from any source/query dataset yet.
    t = torch.arange(256,dtype=torch.float32)
    smoke = torch.stack((torch.cos(t/7),torch.sin(t/7)),0).unsqueeze(0)
    output = reader.classify_features(reader.features(smoke))
    if output.shape!=(1,6) or not bool(torch.isfinite(output).all()): raise ValueError('Smoke failed')
    smoke_sinc = torch.autograd.grad(F.cross_entropy(output,torch.zeros(1,dtype=torch.long)),
                                    [sinc.low_coordinate,sinc.band_coordinate],allow_unused=True)
    if any(g is None or not bool(torch.isfinite(g).all()) for g in smoke_sinc):
        raise ValueError('Shared Sinc gradient disconnected in real checkpoint smoke')
    labeled = SourceDataset(cfg['source_cache'],'L_s',contract)
    validation = SourceDataset(cfg['source_cache'],'V',contract)
    if len(labeled)!=6300 or len(validation)!=27000: raise ValueError('Unexpected source role size')
    li,vi=select_cells(labeled),select_cells(validation)
    rows=[];started=time.monotonic(); core=reader.core.id_backbone
    for repeat,(lidx,vidx) in enumerate(zip(li,vi)):
        lx,ly,lrx,lday=batch(labeled,lidx)
        with capture(core,reader.sequence_pool,reader.response_projection) as feats:
            logits = reader.classify_features(reader.features(lx))
        ce = F.cross_entropy(logits,ly,label_smoothing=.01)
        base = grads(ce,params,retain=True)
        aux = {}
        losses = {}
        for i,name in enumerate(BRANCHES):
            loss=relation_loss(feats[name],ly,lrx,lday)
            losses[name]=float(loss.detach())
            aux[name]=grads(loss,params,retain=i<len(BRANCHES)-1)
        combined=[]
        for i in range(len(params)):
            present=[aux[name][i] for name in BRANCHES if aux[name][i] is not None]
            combined.append(sum(present[1:],present[0].clone()) if present else None)
        aux['all_branches']=combined
        vx,vy,vrx,_=batch(validation,vidx)
        vlogits=reader.classify_features(reader.features(vx))
        vgrads={}
        for i,r in enumerate(RXS):
            mask=vrx==r
            if int(mask.sum())!=18: raise ValueError('Validation RX batch is not balanced')
            vloss=F.cross_entropy(vlogits[mask],vy[mask],label_smoothing=.01)
            vgrads[r]=grads(vloss,params,retain=i<len(RXS)-1)
        branches={name:metrics(g,base,vgrads) for name,g in aux.items()}
        rows.append(dict(ordinal=ORDINALS[repeat], labeled_count=len(lx), validation_count=len(vx),
            raw_labeled_ce=float(ce.detach()), raw_validation_ce=float(F.cross_entropy(vlogits,vy,label_smoothing=.01).detach()),
            loss_by_branch=losses,branches=branches))
        print('SOURCE_BATCH',repeat+1,'/',len(ORDINALS),flush=True)
    if any(p.grad is not None or not torch.equal(p.detach(),old) for p,old in zip(params,before)):
        raise ValueError('Frozen parameter or .grad changed')
    result=dict(status='SOURCE_ONLY_DIAGNOSTIC_COMPLETE',run_id=RUN,parent_run=PARENT.name,
        model_row=ROW,model_seed=2026092701,model_epoch=50,model_mode='eval, parameter gradients measured without update',
        L_loaded=6300,V_loaded=27000,L_forward=1080,V_forward=1080,batches=12,batch_size=90,
        ordinals=list(ORDINALS), source_rxs=list(RXS),source_days=[1,2,3],
        labels='true L and independent V source labels only',
        base='raw clean CE with label_smoothing .01 only; proxy, not full Tri-SIG-TIME training L objective',
        branch_loss='same TX/day different RX positives; same RX/day different TX negatives; tau .2, weight .01 per branch',
        protection='project out negative L-CE gradient component only within actual branch gradient support; classifier untouched',
        validation='five source RX V CE gradient dot products; independent records, same known RX, first-order SGD directional proxy only',
        rows=rows,optimizer_updates=0,target_reads=0,U_reads=0,model_parameters_unchanged=True,
        limitations='One checkpoint, 12 balanced batches, no true train-mode/dropout/As/KL/contrast or U objective; V gradient is for diagnostic evaluation only.',
        seconds=time.monotonic()-started)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (out/'complete.json').write_text(json.dumps(dict(status=result['status'],run_id=RUN,batches=12,L_forward=1080,V_forward=1080,target_reads=0,optimizer_updates=0))+'\n',encoding='utf8')
    print('COMPLETED',json.dumps({'run_id':RUN,'batches':12}),flush=True)


if __name__=='__main__': main()
