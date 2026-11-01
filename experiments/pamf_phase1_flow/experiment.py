"""Source-only flow training and fixed Tri-SIG-TIME Phase1 evaluation."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments.pamf_identity_flow.model import IdentityConditionalFlow
from experiments.pamf_identity_flow.rollout_objective import rollout_objective

RUN = '20261101-phase1-trisigtime-source-rollout-E30-s2026092701-r02'
F_RUN = '20261027-phase1-pamf-repl-shuffled-views-m16-r02'
F_ROW = 'main_split-repl-shuffled_time-3view-s2026092701'
VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
EPOCHS, BATCH, COUNT = 30, 128, 63000
STEPS = EPOCHS * math.ceil(COUNT / BATCH)


class FixedAntitheticFlow(torch.nn.Module):
    """Fixed two-path mean with no target-dependent choices."""
    def __init__(self, flow):
        super().__init__(); self.flow=flow
        generator=torch.Generator(device='cpu').manual_seed(2026110103)
        self.register_buffer('noise_template',torch.randn((1,2,256),generator=generator))

    def forward(self, received):
        noise=self.noise_template.expand_as(received)
        a=self.flow.integrate(received,noise,noise_scale=.02,steps=4)
        b=self.flow.integrate(received,-noise,noise_scale=.02,steps=4)
        return .5*(a+b)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def teacher(root, device):
    from experiments.cric_dura_comparison.bootstrap import activate
    native = root / 'releases' / (F_RUN + '-release')
    runtime = read(native / 'runtime.json')
    if runtime['code_commit'] != '3ff8f4b7e85b5608f5b45350aa18081499a7ef1d':
        raise ValueError('Exact Phase1 Tri-SIG-TIME release required')
    activate(runtime)
    import experiments
    native_experiments=str(native/'experiments')
    if native_experiments not in experiments.__path__:experiments.__path__.append(native_experiments)
    from experiments.pamf_repl_view_shuffle.source import load_reader
    folder = root / 'runs' / F_RUN / 'F' / F_ROW / 'source'
    complete = read(folder / 'complete.json')
    contract = read(complete['config']['source_contract'])
    net, config = load_reader(folder, contract, device)
    if config['row_id'] != F_ROW or sum(p.numel() for p in net.parameters()) != 232993:
        raise ValueError('Exact three-view Tri-SIG-TIME reader required')
    return net.eval().requires_grad_(False), contract


def data(root, contract):
    cache = root / 'runs/20261019-phase1-restoration-learned-action-source-s2026092701-r07/cache'
    done = read(cache / 'complete.json')
    if (done.get('status'), done.get('train_count'), done.get('TX_labels_read'), done.get('target_reads')) != ('ACTION_SOURCE_CACHE_COMPLETE', COUNT, 0, 0):
        raise ValueError('Complete source-only Phase1 cache required')
    clean = np.load(cache / 'train/clean.npy', mmap_mode='r', allow_pickle=False)
    roles = np.load(cache / 'train/roles.npy', allow_pickle=False)
    ids = np.load(cache / 'train/physical_ids.npy', allow_pickle=False)
    if clean.shape != (COUNT, 2, 256) or clean.dtype != np.float32 or roles.shape != (COUNT,) or ids.shape != (COUNT,) or len(set(ids.tolist())) != COUNT:
        raise ValueError('Phase1 aligned source arrays required')
    for role, size in (('L_s', 6300), ('U_s', 56700)):
        if (roles == role).sum() != size or set(ids[roles == role].tolist()) != set(contract['role_ids'][role]):
            raise ValueError('Tri-SIG-TIME and restorer physical source roles differ')
    return cache, clean, roles


def make_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026092701)
        return IdentityConditionalFlow()


def tensor(x, device):
    values = np.array(x, dtype=np.float32, copy=True, order='C')
    return torch.frombuffer(memoryview(values), dtype=torch.float32).reshape(values.shape).to(device)


def save(path, model, commit, epoch):
    with Path(path).open('xb') as f:
        torch.save(dict(run=RUN, commit=commit, epoch=epoch, updates=epoch * math.ceil(COUNT/BATCH),
                        F_run=F_RUN, F_row=F_ROW, source='Phase1 R07 L_s/U_s', model=model.state_dict()), f)


def load(path, device, commit, epoch=EPOCHS):
    data = torch.load(path, map_location='cpu', weights_only=False)
    expected = dict(run=RUN, commit=commit, epoch=epoch, updates=epoch * math.ceil(COUNT/BATCH),
                    F_run=F_RUN, F_row=F_ROW, source='Phase1 R07 L_s/U_s')
    if any(data.get(k) != v for k, v in expected.items()):
        raise ValueError('Wrong fixed Phase1 source checkpoint')
    model = make_model(); model.load_state_dict(data['model'], strict=True)
    return FixedAntitheticFlow(model).to(device).eval().requires_grad_(False)


def smoke(root, out, commit):
    f, contract = teacher(root, 'cpu'); cache, clean, roles = data(root, contract)
    idx = np.flatnonzero(roles == 'L_s')[:2]
    x = tensor(clean[idx], 'cpu')
    y = tensor(np.load(cache/'epochs/01/received.npy', mmap_mode='r', allow_pickle=False)[idx], 'cpu')
    model = make_model(); noise = torch.randn(y.shape, generator=torch.Generator().manual_seed(2026110105))
    before = {k: v.clone() for k, v in f.state_dict().items()}
    loss, terms = rollout_objective(model, f, x, y, noise)
    gradients = [torch.autograd.grad(terms[k].sum(), model.output.weight, retain_graph=True)[0] for k in ('iq','identity')]
    if any(not torch.isfinite(g).all() or not g.abs().sum() > 0 for g in gradients):
        raise ValueError('Both actual source IQ and Tri-SIG feature gradients required')
    loss.mean().backward(); torch.optim.AdamW(model.parameters(), lr=.0003).step()
    if any(p.grad is not None for p in f.parameters()) or any(not torch.equal(v, before[k]) for k,v in f.state_dict().items()):
        raise ValueError('Tri-SIG-TIME must remain frozen')
    save(out/'smoke.pt', model, commit, 0)
    restored = load(out/'smoke.pt', 'cpu', commit, epoch=0)
    with torch.no_grad():
        actual = restored(y); expected = FixedAntitheticFlow(model).eval()(y)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        logits = f.classify_features(f.features(actual))
    if logits.shape != (2,6) or not torch.isfinite(logits).all(): raise ValueError('Nonfinite frozen pipeline')
    write(out/'smoke.json', dict(status='PASS', F_run=F_RUN, F_row=F_ROW, source_rows=2,
          gradient_norms=[float(g.norm()) for g in gradients], F_frozen=True, checkpoint_reload=True,
          TX_labels_read=0, test_IQ_reads=0))


def lr(update):
    if update < 493: return .0003 * (update + 1) / 493
    fraction = (update - 493) / (STEPS - 493 - 1)
    return .00002 + (.0003 - .00002) * .5 * (1 + math.cos(math.pi*fraction))


def train(root, out, commit):
    f, contract = teacher(root, 'cuda:0'); cache, clean, roles = data(root, contract)
    frozen = {k:v.detach().cpu().clone() for k,v in f.state_dict().items()}
    model = make_model().cuda().train(); optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=.0001)
    rng = np.random.default_rng(2026110104); torch.manual_seed(2026110105); torch.cuda.manual_seed_all(2026110105)
    row = out/'flow'; row.mkdir(); updates=0; started=time.perf_counter()
    with (row/'training.jsonl').open('x', encoding='utf-8') as log:
        for epoch in range(1,EPOCHS+1):
            view=(epoch-1)%20+1
            received=np.load(cache/'epochs'/f'{view:02d}'/'received.npy',mmap_mode='r',allow_pickle=False)
            if received.shape != clean.shape or received.dtype != np.float32: raise ValueError('Phase1 source view mismatch')
            order=rng.permutation(COUNT); clean_count=COUNT//4; totals=dict(loss=0.,iq=0.,identity=0.)
            for offset in range(0,COUNT,BATCH):
                ix=order[offset:offset+BATCH]; a=tensor(clean[ix], 'cuda:0'); b=tensor(received[ix], 'cuda:0')
                # A fixed 25% of each randomized epoch is clean identity training.
                n=max(0,min(len(ix),clean_count-offset))
                if n: b[:n]=a[:n]
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups: group['lr']=lr(updates)
                loss,terms=rollout_objective(model,f,a,b,torch.randn_like(a))
                if not torch.isfinite(loss).all(): raise FloatingPointError('Nonfinite training loss')
                loss.mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();updates+=1
                for key, value in dict(loss=loss,iq=terms['iq'],identity=terms['identity']).items(): totals[key]+=float(value.detach().sum())
            record=dict(epoch=epoch,view=view,updates=updates,losses={k:v/COUNT for k,v in totals.items()},seconds=time.perf_counter()-started)
            log.write(json.dumps(record,allow_nan=False)+'\n');log.flush();print(json.dumps(record),flush=True)
    if updates != STEPS or any(p.grad is not None for p in f.parameters()) or any(not torch.equal(v.detach().cpu(),frozen[k]) for k,v in f.state_dict().items()):
        raise ValueError('Training budget or frozen Tri-SIG-TIME invariant failed')
    save(row/'final.pt',model,commit,EPOCHS)
    write(row/'training_complete.json',dict(status='SOURCE_TRAINED',epoch=EPOCHS,updates=updates,F_run=F_RUN,F_row=F_ROW,
          F_frozen=True,TX_labels_read=0,test_IQ_reads=0,source_roles=['L_s','U_s'],parameters=sum(p.numel() for p in model.parameters())))


def reference(root):
    folder=root/'runs/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/reference'
    m=read(folder/'manifest.json'); ids=np.load(folder/'physical_ids.npy',allow_pickle=False)
    if m.get('status')!='FORMAL_REFERENCE_READY' or m.get('count')!=168000 or m.get('views')!=list(VIEWS) or m.get('TX_truth_reads')!=0 or ids.shape!=(168000,) or len(set(ids.tolist()))!=len(ids):
        raise ValueError('Exact complete Phase1 ground test reference required')
    return folder,ids


@torch.no_grad()
def predict(root,out,commit):
    done=read(out/'flow/training_complete.json')
    if (done['status'],done['epoch'],done['updates'],done['F_run'],done['F_row'],done['test_IQ_reads']) != ('SOURCE_TRAINED',EPOCHS,STEPS,F_RUN,F_ROW,0): raise ValueError('Fixed source training required')
    model=load(out/'flow/final.pt','cuda:0',commit);f,_=teacher(root,'cuda:0');folder,ids=reference(root)
    dst=out/'flow/test_prediction';dst.mkdir();np.save(dst/'physical_ids.npy',ids,allow_pickle=False)
    for view in VIEWS:
        values=np.load(folder/(view+'.npy'),mmap_mode='r',allow_pickle=False)
        if values.shape!=(len(ids),2,256) or values.dtype!=np.float32: raise ValueError('Phase1 received view mismatch')
        raw=np.lib.format.open_memmap(dst/(view+'.raw.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        restored=np.lib.format.open_memmap(dst/(view+'.restored.npy'),mode='w+',dtype=np.int64,shape=(len(ids),))
        for offset in range(0,len(ids),BATCH):
            x=tensor(values[offset:offset+BATCH],'cuda:0'); y=model(x)
            a=f.classify_features(f.features(x));b=f.classify_features(f.features(y))
            if not torch.isfinite(y).all() or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite fixed prediction')
            raw[offset:offset+len(x)]=np.from_dlpack(a.argmax(1).cpu());restored[offset:offset+len(x)]=np.from_dlpack(b.argmax(1).cpu())
        raw.flush();restored.flush();print(json.dumps(dict(view=view,count=len(ids))),flush=True)
    write(dst/'complete.json',dict(status='PREDICTIONS_COMPLETE',count=len(ids),views=list(VIEWS),F_run=F_RUN,F_row=F_ROW,
          commit=commit,TX_truth_reads=0,model_updates=0))


def validate_prediction_manifest(done, *, count, commit):
    expected=dict(status='PREDICTIONS_COMPLETE',count=count,views=list(VIEWS),F_run=F_RUN,F_row=F_ROW,
                  commit=commit,TX_truth_reads=0,model_updates=0)
    if not isinstance(done,dict) or any(done.get(k)!=v for k,v in expected.items()) or set(done)-set(expected)-{'accepted_by_view'}:
        raise ValueError('All predictions required before truth')
    if 'accepted_by_view' in done:
        counts=done['accepted_by_view']
        if not isinstance(counts,dict) or set(counts)!=set(VIEWS) or any(type(v) is not int or v<0 or v>count for v in counts.values()):
            raise ValueError('Valid per-view accepted counts required')


def score(root,out,commit):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU truth scorer required')
    folder,ids=reference(root);dst=out/'flow/test_prediction';done=read(dst/'complete.json')
    validate_prediction_manifest(done,count=len(ids),commit=commit)
    if not np.array_equal(ids,np.load(dst/'physical_ids.npy',allow_pickle=False)):raise ValueError('IDs differ')
    pred={}
    for view in VIEWS:
        pred[view]={}
        for arm in ('raw','restored'):
            a=np.load(dst/(view+'.'+arm+'.npy'),allow_pickle=False)
            if a.shape!=(len(ids),) or a.dtype!=np.int64 or np.any((a<0)|(a>=6)):raise ValueError('Incomplete predictions')
            pred[view][arm]=a
    truth=read(root/'runs/20260927-phase1-baselines-final-clean-satellite-m5-r01/data/truth.json')
    labels=np.asarray([truth[str(i)]['label'] for i in ids]);rx=np.asarray([str(truth[str(i)]['receiver']) for i in ids]);day=np.asarray([str(truth[str(i)].get('day','unavailable')) for i in ids])
    clean=float(np.mean(pred['clean']['raw']==labels)*100);results={};groups=[]
    for view in VIEWS:
        a=pred[view]['raw'];b=pred[view]['restored'];acc=float(np.mean(b==labels)*100)
        results[view]=dict(raw_accuracy=float(np.mean(a==labels)*100),restored_accuracy=acc,raw_clean_gap_pp=clean-acc,
                           macro_f1=float(np.mean([2*np.sum((b==i)&(labels==i))/max(1,np.sum(b==i)+np.sum(labels==i)) for i in range(6)])*100))
        for name,vec in (('TX',labels),('RX',rx),('day',day)):
            for value in np.unique(vec):
                mask=vec==value;groups.append(dict(view=view,dimension=name,value=str(value),count=int(mask.sum()),raw=float(np.mean(a[mask]==labels[mask])*100),restored=float(np.mean(b[mask]==labels[mask])*100)))
    write(out/'flow/test_summary.json',dict(status='ANALYZED',count=len(ids),F_run=F_RUN,F_row=F_ROW,raw_clean_accuracy=clean,
          views=results,groups=groups,six_scene_gap_at_most_3pp=all(results[v]['raw_clean_gap_pp']<=3 for v in VIEWS[1:]),
          clean_damage_at_most_1pp=results['clean']['raw_clean_gap_pp']<=1,all_predictions_before_truth=True,target_feedback=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','train','predict','score'));p.add_argument('--root',type=Path,required=True);p.add_argument('--commit',required=True);a=p.parse_args()
    if Path('release_commit.txt').read_text(encoding='ascii').strip()!=a.commit:raise ValueError('Pinned release required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=a.root/'runs'/RUN
    {'smoke':smoke,'train':train,'predict':predict,'score':score}[a.mode](a.root,out,a.commit)


if __name__=='__main__':main()
