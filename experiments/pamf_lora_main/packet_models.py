"""Tri packet aggregation, source-calibrated SSL, and bounded DURA-LoRa."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.cric_dura_comparison.io import read, tensor, write
from .comparison import dura_model, seed_all
from .data import VIEWS
from .packet_data import CHIRPS, SINGLE_INDEX
from .physical import ChirpResponse


SEED = 2026092701
METHODS = ('single_sup', 'packet_sup', 'packet_pl', 'packet_tri')
EPOCHS = 30
BATCH = 32
G_EPOCHS = 15


class PacketTri(nn.Module):
    def __init__(self, model, views):
        super().__init__()
        self.encoder = model
        self.views = views

    def forward(self, x):
        if self.views == 1:
            x = x[:, SINGLE_INDEX:SINGLE_INDEX+1]
        fused = self.encoder.features(x.flatten(0, 1)).reshape(len(x), self.views, 160).mean(dim=1)
        return self.encoder.classify_features(fused), fused


def packet_tri(method, device):
    from experiments.cric_dura_comparison import f_train, f_model
    from experiments.cvs_phase1_stack.runtime import installed
    from .recognition import config, reader_scope
    c = config(dict(ssl={}, run_id='lora_packet', code_commit='', source_cache='',
                    source_contract='', source_practical=''), 'pamf_sup', SEED, Path(method))
    c['paper_row']['permuted'] = True
    original = f_train.original_config(SEED)
    with f_train.class_dimension(c), installed(original) as native, reader_scope(c):
        args = f_train.arguments(c, device)
        args.input_len = 256
        model = f_model.install(native.build_baseline_model(args, device), c).id_backbone.encoder
    model.response = ChirpResponse(fft_factor=4)
    return PacketTri(model, 1 if method == 'single_sup' else len(CHIRPS)).to(device)


def sym_kl(a, b):
    a, b = a.log_softmax(1), b.log_softmax(1)
    return (.5*(a.exp()-b.exp())*(a-b)).sum(1)


def cross_rx(z, y, rx, *, balance=False):
    same_y = y[:, None] == y
    same_rx = rx[:, None] == rx
    positives = same_y & ~same_rx
    valid = positives.any(1) & (~same_y).any(1)
    scores = F.normalize(z, dim=1) @ F.normalize(z, dim=1).T / .2
    included = positives | ~same_y
    scores = scores.masked_fill(~included, -1e4).log_softmax(1)
    uniform = positives / positives.sum(dim=1, keepdim=True).clamp_min(1)
    if balance:
        support = positives.float() @ same_rx.float()
        by_rx = positives / support.clamp_min(1)
        target = .5 * (uniform + by_rx / by_rx.sum(dim=1, keepdim=True).clamp_min(1))
    else:
        target = uniform
    return (-(scores * target).sum(1) * valid).sum() / valid.sum().clamp_min(1), valid.sum()


@torch.no_grad()
def choose_threshold(logits, y, rx):
    confidence, pred = logits.softmax(1).max(1)
    candidates = []
    for t in (.65,.75,.85,.95):
        mask = confidence >= t
        n = int(mask.sum())
        hit = int((mask & (pred==y)).sum())
        p = hit/max(1,n)
        w = 1.96
        lower = (p + w*w/(2*max(1,n)) - w*math.sqrt(
            (p*(1-p)+w*w/(4*max(1,n)))/max(1,n))) / (1+w*w/max(1,n))
        support = len(rx[mask].unique())
        coverage_classes = len(pred[mask].unique())
        eligible = n >= 24 and lower >= .75 and support >= 4 and coverage_classes >= 4
        candidates.append(dict(t=t, accepted=n, correct=hit, lower=lower, receivers=support,
                               classes=coverage_classes, eligible=eligible))
    chosen = next((e for e in candidates if e['eligible']), None)
    return dict(threshold=chosen['t'] if chosen else 1.,
                U_weight=.75*chosen['lower'] if chosen else 0., candidates=candidates)


@torch.no_grad()
def val(model, x, y, batch=64):
    model.eval()
    a = torch.cat([model(v)[0] for v in x.split(batch)])
    return dict(accuracy=float((a.argmax(1)==y).float().mean()), ce=float(F.cross_entropy(a,y))), a


def load_source_labels(root, role):
    if role not in ('L_s','V'):
        raise ValueError('Only labeled source roles may be read')
    with np.load(root/f'{role}_metadata.npz', allow_pickle=False) as m:
        c = read(root/'contract.json')
        if m['ids'].tolist() != c['role_ids'][role]:
            raise ValueError('Packet label order differs')
        return tensor(m['y'],'cuda:0'), tensor(m['rx_i'],'cuda:0')


def train_f(source, cache, out, method):
    if method not in METHODS or read(source/'contract.json')['protocol'] != 'lora_packet_pilot_v1':
        raise ValueError('Unknown LoRa F source row')
    out.mkdir(parents=True, exist_ok=False)
    seed_all(SEED)
    x, v = (tensor(np.load(cache/f'{role}.npy', allow_pickle=False),'cuda:0') for role in ('L_s','V'))
    y, rx = load_source_labels(source,'L_s')
    vy, vr = load_source_labels(source,'V')
    use_u = method in ('packet_tri','packet_pl')
    if use_u:
        u = tensor(np.load(cache/'U_s.npy',allow_pickle=False)[0],'cuda:0')
        with np.load(source/'U_s_metadata.npz', allow_pickle=False) as m:
            if 'y' in m.files or m['ids'].tolist() != read(source/'contract.json')['role_ids']['U_s']:
                raise ValueError('U label permission or packet order differs')
            ur = tensor(m['rx_i'],'cuda:0')
    model = packet_tri(method,'cuda:0')
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    steps = math.ceil(len(y)/BATCH)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps*EPOCHS, eta_min=2e-5)
    policy = dict(threshold=1.,U_weight=0.)
    with (out/'epochs.jsonl').open('x',encoding='utf-8') as log:
        import json
        for epoch in range(1,EPOCHS+1):
            if epoch >= 11 and (epoch-11)%5==0 and method=='packet_tri':
                _, logits = val(model, v[0], vy)
                policy = choose_threshold(logits,vy,vr)
            elif epoch >= 11 and method=='packet_pl':
                policy = dict(threshold=.95,U_weight=1.)
            seen = accepted = anchors = 0
            averages = torch.zeros(4,device='cuda:0')
            model.train()
            for idx in torch.randperm(len(y),device='cuda:0').split(BATCH):
                scene = torch.randint(1,len(VIEWS),(len(idx),),device='cuda:0')
                a,b = x[0,idx],x[scene,idx]
                score, z = model(torch.cat((a,b)))
                raw, sat = score.chunk(2)
                layout, _ = cross_rx(z[:len(idx)],y[idx],rx[idx])
                ce, ce_sat = F.cross_entropy(raw,y[idx]), F.cross_entropy(sat,y[idx])
                base = ce + .68*ce_sat + .10*layout + .10*sym_kl(raw,sat).mean()
                loss_u = base.new_zeros(())
                if use_u and policy['U_weight']:
                    ui = torch.randint(len(u),(BATCH,),device='cuda:0')
                    model.eval()
                    with torch.no_grad():
                        conf,pseudo = model(u[ui])[0].softmax(1).max(1)
                    model.train()
                    allow = conf >= policy['threshold']
                    count = int(allow.sum())
                    seen += len(ui)
                    accepted += count
                    if count:
                        inp, yp = u[ui[allow]], pseudo[allow]
                        student, zu = model(inp + .015*torch.randn_like(inp))
                        dose = policy['U_weight']
                        if method=='packet_tri':
                            aux, used = cross_rx(zu,yp,ur[ui[allow]],balance=True)
                            anchors += int(used)
                            loss_u = dose * ((F.cross_entropy(student,yp,reduction='sum')/max(16,count)) + .05*aux)
                        else:
                            loss_u = dose * F.cross_entropy(student,yp,reduction='sum')/len(ui)
                opt.zero_grad(set_to_none=True)
                loss = base + loss_u
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step(); sch.step()
                averages += torch.stack((ce.detach(),ce_sat.detach(),layout.detach(),loss_u.detach()))
            summary, logits = val(model,v[0],vy)
            per_scene = {}
            if epoch%5==0:
                per_scene = {name:val(model,v[i],vy)[0]['accuracy'] for i,name in enumerate(VIEWS) if i}
            by_rx = [float((logits[vr==i].argmax(1)==vy[vr==i]).float().mean()) for i in range(5)]
            losses = dict(zip(('clean_ce','practical_ce','cross_rx','U_applied'),(averages/steps).tolist(),strict=True))
            record = dict(epoch=epoch, losses=losses, source=summary, by_rx=by_rx,
                          source_practical=per_scene, U_policy=policy, U_seen=seen, U_accepted=accepted, U_anchors=anchors)
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
            print('PACKET_F_EPOCH',method,json.dumps(record,allow_nan=False),flush=True)
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'complete.json',dict(status='SOURCE_FROZEN',method=method,seed=SEED,epochs=EPOCHS,
          initial_checkpoint=None,target_read=False,source_contract=read(source/'contract.json')))


class GuardedPacketDura(nn.Module):
    """Use observed chirp coherence to preserve clean packets exactly."""
    def __init__(self):
        super().__init__()
        self.net = dura_model()
        self.reference = ChirpResponse(fft_factor=8).requires_grad_(False)

    def forward(self, x, *, with_gate=False):
        frames = x.flatten(0,1)
        with torch.no_grad():
            quality = self.reference.coherence(frames).reshape(len(x),len(CHIRPS)).median(dim=1).values
            gate = ((.94-quality)/.24).clamp(0.,1.)
        output = x.clone()
        active = gate > 0.
        if bool(active.any()):
            received = x[active].flatten(0,1)
            full = self.net.stage_outputs(received)[0][-1]
            residual = (full-received).reshape(-1,len(CHIRPS),2,256)
            norm = residual.square().mean((1,2,3)).clamp_min(1e-8).sqrt()
            budget = .15*gate[active]*x[active].square().mean((1,2,3)).sqrt()
            scale = (budget/norm).clamp_max(1.)
            output[active] = x[active]+residual*scale[:,None,None,None]
        return (output,gate) if with_gate else output


def train_g(cache, out, source_path):
    out.mkdir(parents=True,exist_ok=False)
    seed_all(SEED)
    raw = tensor(np.load(cache/'G_pairs.npy',allow_pickle=False),'cuda:0')
    model = GuardedPacketDura().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=G_EPOCHS*math.ceil(len(raw[0])/BATCH),eta_min=2e-5)
    import json
    with (out/'epochs.jsonl').open('x',encoding='utf-8') as log:
        for epoch in range(1,G_EPOCHS+1):
            loss_sum = identity = exposed = 0
            for i in torch.randperm(len(raw[0]),device='cuda:0').split(BATCH):
                ground, received = raw[:,i]
                out_i, gate = model(received,with_gate=True)
                identity += int((gate==0).sum())
                # Quotient only global phase during training so numerical phase
                # disagreement does not reward overwriting within-chirp detail.
                z = torch.complex(out_i[:,:,0],out_i[:,:,1])
                c = torch.complex(ground[:,:,0],ground[:,:,1])
                phase = (z.conj()*c).sum((1,2)).detach()
                unit = phase / phase.abs().clamp_min(1e-8)
                loss = (z*unit[:,None,None]-c).abs().square().mean() + .10*(out_i-received).square().mean()
                opt.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                    opt.step()
                sch.step()
                loss_sum += float(loss.detach())*len(i); exposed += len(i)
            row = dict(epoch=epoch, loss=loss_sum/exposed, pass_through=identity, exposed=exposed)
            log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush()
            print('PACKET_G_EPOCH',json.dumps(row),flush=True)
    torch.save(model.state_dict(),out/'model.pt')
    write(out/'complete.json',dict(status='SOURCE_FROZEN',epochs=G_EPOCHS,seed=SEED,TX_labels_read=False,
          F_reads=False,target_read=False,source_contract=read(source_path/'contract.json')))
