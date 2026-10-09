"""Source receiver *distribution* actions. No paired-RX counterfactual claim.

All nonlinear maps are evaluated per packet before reduction. Descriptors,
recipients and destinations use disjoint physical bags and descriptors exclude
the recipient TX. One parameter set is fitted jointly over input conditions.
"""
from collections import defaultdict
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from experiments.cvs_multi_action_audit.receiver import (
    ContributionStatistics, packet_time_statistics, ReceiverAction, _fixed,
)
from experiments.cvs_multi_disentangle.model import (
    FixedViews, intermediate, identity_from_intermediate, classify_intermediate,
)


def margin_vector(logits, y):
    y = torch.as_tensor(y, device=logits.device, dtype=torch.long).expand(len(logits))
    mask = torch.arange(logits.shape[1], device=logits.device)[None] != y[:, None]
    return (logits.gather(1, y[:, None]) - logits)[mask].reshape(len(logits), -1)


class ReceiverDistributionAction(nn.Module):
    """Shared receiver code; packet-dependent action with exact zero identity."""
    def __init__(self, width=49, dimension=349):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 8))
        self.state = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 64), nn.SiLU())
        self.decoder = nn.Sequential(nn.Linear(72, 64), nn.SiLU(), nn.Linear(64, dimension))
        nn.init.normal_(self.decoder[-1].weight, std=.001)
        nn.init.zeros_(self.decoder[-1].bias)

    def code(self, v0, v1):
        return self.encoder(v1) - self.encoder(v0)

    def forward(self, h, v0, v1):
        q = self.code(v0, v1).expand(len(h), -1)
        s = self.state(h)
        return self.decoder(torch.cat((s, q), 1)) - self.decoder(torch.cat((s, torch.zeros_like(q)), 1))


class OnlineReceiver(nn.Module):
    """Versioned source-only packet contributions for online distribution fits.

    Reobserving an ID does not rejuvenate its original contribution. A reference
    refresh clears the pool. No IQ is retained. The caller owns optimizer steps.
    """
    def __init__(self, max_age_steps=888, action=None):
        super().__init__()
        self.action = action if action is not None else ReceiverDistributionAction()
        self.max_age_steps = int(max_age_steps)
        self.entries = {}
        self.reference_version = None
        self.expired = 0
        self.counter = 0
        self.live_calibration = {}

    def reset(self, version):
        self.entries.clear()
        self.reference_version = str(version)
        self.live_calibration.clear()

    def expire(self, step, version):
        if self.reference_version != str(version):
            self.reset(version)
        for key in list(self.entries):
            age = int(step)-self.entries[key]['step']
            if age < 0:
                raise ValueError('Receiver contribution clock moved backwards')
            if age > self.max_age_steps:
                del self.entries[key]; self.expired += 1

    def update(self, x, h, y, rx, day, ids, condition, step, version, reference_identity):
        self.expire(step, version)
        fields = validate_data(dict(x=x,y=y,rx=rx,day=day,ids=ids,condition=condition))
        with torch.no_grad():
            view = FixedViews().to(x.device).eval()
            v = torch.cat((view.receiver(x),packet_time_statistics(x)),1)
            z = identity_from_intermediate(reference_identity,h)
            logits = classify_intermediate(reference_identity,h)
        for i,pid in enumerate(fields['ids']):
            key=(pid,fields['condition'][i])
            if key in self.entries:
                continue
            role=int(hashlib.sha256(str(pid).encode('utf-8')).hexdigest(),16)%3
            self.entries[key]=dict(pid=pid,condition=fields['condition'][i],
                y=int(fields['y'][i]),rx=int(fields['rx'][i]),day=int(fields['day'][i]),
                step=int(step),role=role,h=h[i].detach(),v=v[i].detach(),
                z=z[i].detach(),logits=logits[i].detach())
        return dict(contributions=len(self.entries),expired=self.expired,reference_version=str(version))

    def _groups(self, step, version, partition=0):
        self.expire(step,version)
        groups=defaultdict(lambda:[[],[],[]])
        for entry in self.entries.values():
            if (int(hashlib.sha256(str(entry['pid']).encode('utf-8')).hexdigest(),16)//3)%2 != partition:
                continue
            groups[(entry['y'],entry['rx'],entry['day'],entry['condition'])][entry['role']].append(entry)
        return groups

    def _contexts(self, groups, source_key):
        y,rx,day,condition=source_key
        for dest_key,roles in sorted(groups.items(),key=lambda kv:str(kv[0])):
            if dest_key[0]!=y or dest_key[1]==rx or dest_key[2:]!=(day,condition) or not roles[2]:
                continue
            donors=[k for k,bags in groups.items() if k[0]!=y and k[1:]==(rx,day,condition)
                and bags[0] and (k[0],dest_key[1],day,condition) in groups
                and groups[(k[0],dest_key[1],day,condition)][0]]
            if not donors:continue
            v0=torch.stack([torch.stack([e['v'] for e in groups[k][0]]).mean(0) for k in donors]).mean(0)[None]
            v1=torch.stack([torch.stack([e['v'] for e in groups[(k[0],dest_key[1],day,condition)][0]]).mean(0) for k in donors]).mean(0)[None]
            anchors=torch.stack([F.normalize(torch.stack([e['z'] for e in bags[2]]),dim=1).mean(0)
                for k,bags in sorted(groups.items(),key=lambda kv:str(kv[0])) if k[1:]==dest_key[1:] and bags[2]])
            yield dest_key,roles[2],v0,v1,anchors

    @staticmethod
    def _loss(identity,moved,y,dest,anchors):
        pred=packet_statistics(identity,moved,y)
        z=torch.stack([e['z'] for e in dest]).detach()
        logits=torch.stack([e['logits'] for e in dest]).detach()
        target=dict(z=z.mean(0),normz=F.normalize(z,dim=1).mean(0),margin=margin_vector(logits,y).mean(0))
        direction=F.normalize(anchors.detach(),dim=1)
        return ((pred['z']-target['z']).square().mean()
            +(pred['normz']-target['normz']).square().mean()
            +.1*(pred['margin']-target['margin']).square().mean()
            +.1*((pred['normz']-target['normz'])@direction.T).square().mean())

    def fit_loss(self, reference_identity, step, version):
        groups=self._groups(step,version)
        contexts=[(key,bags[1],context) for key,bags in sorted(groups.items(),key=lambda kv:str(kv[0]))
            if bags[1] for context in self._contexts(groups,key)]
        if not contexts:return None,dict(active=False,reason='insufficient_disjoint_day_matched_bags')
        key,source,(_,dest,v0,v1,anchors)=contexts[self.counter%len(contexts)];self.counter+=1
        h=torch.stack([e['h'] for e in source[:32]])
        # Deterministically bounded descriptor scaling; no audit-derived moments.
        moved=h+self.action(h,v0/8,v1/8)
        loss=self._loss(reference_identity,moved,key[0],dest,anchors)
        return loss,dict(active=True,relations=len(contexts),recipient_packets=len(h),condition=key[3])

    @torch.no_grad()
    def calibrate(self, reference_identity, step, version, max_relations=32):
        """Hash-held-out physical packets; no gradients, condition reliability."""
        groups=self._groups(step,version,partition=1)
        metrics=defaultdict(list)
        contexts=[(key,bags[1],context) for key,bags in sorted(groups.items(),key=lambda kv:str(kv[0]))
            if bags[1] for context in self._contexts(groups,key)]
        if not contexts:return dict(active=False,weights={},reason='insufficient_heldout_receiver_bags')
        # Cycle over all strata so no receiver/class is favored by lexical order.
        start=self.counter%len(contexts)
        selected=(contexts[start:]+contexts[:start])[:max_relations]
        for key,source,(_,dest,v0,v1,anchors) in selected:
            h=torch.stack([e['h'] for e in source[:32]])
            moved=h+self.action(h,v0/8,v1/8)
            learned=self._loss(reference_identity,moved,key[0],dest,anchors)
            zero=self._loss(reference_identity,h,key[0],dest,anchors)
            # Independent directed other-TX mean, same heldout descriptor role.
            destination_rx=dest[0]['rx']
            donors=[k for k,bags in groups.items() if k[0]!=key[0] and k[1:]==key[1:]
                and bags[0] and (k[0],destination_rx,key[2],key[3]) in groups
                and groups[(k[0],destination_rx,key[2],key[3])][0]]
            mean_delta=torch.stack([torch.stack([e['h'] for e in groups[(k[0],destination_rx,key[2],key[3])][0]]).mean(0)
                -torch.stack([e['h'] for e in groups[k][0]]).mean(0) for k in donors]).mean(0)
            strong=self._loss(reference_identity,h+mean_delta,key[0],dest,anchors)
            metrics[key[3]].append((float(learned),float(zero),float(strong)))
        rows={c:dict(relations=len(v),learned=sum(a for a,_,_ in v)/len(v),
            zero=sum(b for _,b,_ in v)/len(v),directed_other_tx_mean=sum(b for _,_,b in v)/len(v))
            for c,v in metrics.items()}
        weights={c:max(0.,min(1.,1-r['learned']/max(min(r['zero'],r['directed_other_tx_mean']),1e-12))) for c,r in rows.items()}
        return dict(active=True,weights=weights,metrics=rows,physical_holdout=True,
            reference_version=str(version),scope='group_condition_reliability_not_packetwise')

    def identity_loss(self, identity, h, y, rx, day, ids, condition, step, version):
        groups=self._groups(step,version)
        n=len(h); ys,rxs,days,conditions=[_values(v,n) for v in (y,rx,day,condition)]
        ids=_values(ids,n)
        grouped=defaultdict(list)
        for i,(yy,rr,dd,cc) in enumerate(zip(ys,rxs,days,conditions)):
            # Only live recipient-role IDs are accepted; donor/target IDs cannot
            # simultaneously act as recipients, even across condition views.
            pid=ids[i]
            if int(hashlib.sha256(str(pid).encode('utf-8')).hexdigest(),16)%3==1:
                grouped[(int(yy),int(rr),int(dd),cc)].append(i)
        losses=[]
        for key,ix in grouped.items():
            for _,dest,v0,v1,anchors in self._contexts(groups,key):
                with torch.no_grad():delta=self.action(h[ix].detach(),v0/8,v1/8)
                losses.append(self._loss(identity,h[ix]+delta,key[0],dest,anchors))
        if not losses:return None,dict(active=False,reason='no_live_recipient_with_matched_destination')
        return torch.stack(losses).mean(),dict(active=True,relations=len(losses),live_packets=n)

    @torch.no_grad()
    def calibrate_live(self, identity, h, y, rx, day, ids, condition, step, version,
                       min_distinct_ids=16, ema_decay=.9):
        """Current student coordinates/G, heldout physical recipients only.

        Targets remain frozen-reference destination *distributions*: this is
        teacher-distribution distillation, not a same-coordinate transport
        assertion. A positive teacher-only score cannot override a harmful
        current-student action. Distinct recipient counts never count repeats.
        """
        groups=self._groups(step,version,partition=1)
        n=len(h)
        ys,rxs,days,conditions,pids=[_values(v,n) for v in (y,rx,day,condition,ids)]
        current=defaultdict(list); excluded=0
        for i,(yy,rr,dd,cc,pid) in enumerate(zip(ys,rxs,days,conditions,pids)):
            digest=int(hashlib.sha256(str(pid).encode('utf-8')).hexdigest(),16)
            if digest%3!=1 or (digest//3)%2!=1:
                excluded+=1;continue
            current[(int(yy),int(rr),int(dd),cc)].append(i)
        observations=defaultdict(list); observed_ids=defaultdict(set)
        for key,ix in current.items():
            for dest_key,dest,v0,v1,anchors in self._contexts(groups,key):
                live=h[ix].detach()
                moved=live+self.action(live,v0/8,v1/8)
                donors=[k for k,bags in groups.items() if k[0]!=key[0] and k[1:]==key[1:]
                    and bags[0] and (k[0],dest_key[1],key[2],key[3]) in groups
                    and groups[(k[0],dest_key[1],key[2],key[3])][0]]
                mean_delta=torch.stack([torch.stack([e['h'] for e in groups[(k[0],dest_key[1],key[2],key[3])][0]]).mean(0)
                    -torch.stack([e['h'] for e in groups[k][0]]).mean(0) for k in donors]).mean(0)
                teacher_z=torch.stack([e['z'] for e in dest])
                target=dict(normz=F.normalize(teacher_z,dim=1).mean(0),
                    margin=margin_vector(torch.stack([e['logits'] for e in dest]),key[0]).mean(0))
                row={}
                for mode,value in (('learned',moved),('zero',live),('directed_other_tx_mean',live+mean_delta)):
                    pred=packet_statistics(identity,value,key[0])
                    for space in ('normz','margin'):
                        row[mode+'_'+space]=float((pred[space]-target[space]).square().mean())
                observations[key[3]].append(row)
                observed_ids[key[3]].update(str(pids[i]) for i in ix)
        for c,rows in observations.items():
            means={k:sum(row[k] for row in rows)/len(rows) for k in rows[0]}
            state=self.live_calibration.setdefault(c,dict(ids=set(),ema={},updates=0))
            state['ids'].update(observed_ids[c]);state['updates']+=1
            for key,value in means.items():
                state['ema'][key]=value if key not in state['ema'] else ema_decay*state['ema'][key]+(1-ema_decay)*value
        metrics={};weights={}
        for c,state in self.live_calibration.items():
            skills={}
            for space in ('normz','margin'):
                e=state['ema'];strong=min(e['zero_'+space],e['directed_other_tx_mean_'+space])
                skills[space]=1-e['learned_'+space]/max(strong,1e-12)
            enough=len(state['ids'])>=min_distinct_ids
            weights[c]=max(0.,min(1.,*skills.values())) if enough else 0.
            metrics[c]=dict(distinct_recipient_ids=len(state['ids']),updates=state['updates'],
                sufficient_distinct_ids=enough,skills_vs_stronger_baseline=skills,**state['ema'])
        return dict(active=bool(observations),weights=weights,metrics=metrics,
            counts={c:len(s['ids']) for c,s in self.live_calibration.items()},
            excluded_nonheldout_or_nonrecipient_packets=excluded,physical_holdout=True,
            min_distinct_recipient_ids=min_distinct_ids,reference_version=str(version),
            semantics='Current student packet statistics vs frozen-reference destination distribution; teacher-distribution distillation',
            combination='Runtime must use min(frozen-reference reliability, current-student reliability)')


def _values(value, count):
    if isinstance(value, str):
        return [value] * count
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else list(value)


def validate_data(data):
    n = len(data['x'])
    fields = {k: _values(data[k], n) for k in ('y', 'rx', 'day', 'ids', 'condition')}
    if any(len(v) != n for v in fields.values()):
        raise ValueError('Receiver metadata length mismatch')
    if any(int(v) < 0 for k in ('y', 'rx', 'day') for v in fields[k]):
        raise ValueError('Only visible source TX/RX/day permitted')
    keys = list(zip(fields['ids'], fields['condition']))
    if len(set(keys)) != n:
        raise ValueError('Duplicate physical ID within condition')
    metadata = {}
    for pid, y, rx, day in zip(fields['ids'], fields['y'], fields['rx'], fields['day']):
        current = (y, rx, day)
        if pid in metadata and metadata[pid] != current:
            raise ValueError('Physical ID has inconsistent TX/RX/day metadata')
        metadata[pid] = current
    return fields


def make_relations(data, repeat=0):
    """Three bag roles, matched exactly on day and condition, LOTO code.

    Repetitions permute the physical split reproducibly. The same physical
    packet in different condition views retains its bag role within a repeat.
    """
    fields = validate_data(data)
    grouped = defaultdict(list)
    for i, key in enumerate(zip(fields['y'], fields['rx'], fields['day'], fields['condition'])):
        grouped[key].append(i)
    bags = {}
    for key, indices in grouped.items():
        if len(indices) < 3:
            continue
        ordered = sorted(indices, key=lambda i: hashlib.sha256(
            f'{repeat}:{fields["ids"][i]}'.encode('utf-8')).hexdigest())
        bags[key] = [ordered[j::3] for j in range(3)]
    relations = []
    for source in sorted(bags, key=str):
        y, rx, day, condition = source
        for dest in sorted(bags, key=str):
            if dest[0] != y or dest[1] == rx or dest[2:] != source[2:]:
                continue
            donors = [key for key in bags if key[0] != y and key[1:] == source[1:]
                      and (key[0], dest[1], day, condition) in bags]
            if not donors:
                continue
            descriptor0 = [i for key in donors for i in bags[key][0]]
            descriptor1 = [i for key in donors for i in bags[(key[0], dest[1], day, condition)][0]]
            indices = [descriptor0 + descriptor1, bags[source][1], bags[dest][2]]
            physical = [set(fields['ids'][i] for i in ix) for ix in indices]
            if any(physical[a] & physical[b] for a, b in ((0, 1), (0, 2), (1, 2))):
                raise ValueError('Descriptor/recipient/destination physical bag overlap')
            relations.append(dict(y=int(y), source_rx=int(rx), destination_rx=int(dest[1]),
                day=int(day), condition=condition, repeat=repeat,
                descriptor0=descriptor0, descriptor1=descriptor1,
                recipient=bags[source][1], destination=bags[dest][2],
                donor_tx=sorted({int(key[0]) for key in donors}),
                donor_groups=[(bags[key][0], bags[(key[0], dest[1], day, condition)][0]) for key in donors]))
    return relations, bags


def _encode(identity, data):
    views = FixedViews().to(data['x'].device).eval()
    hs, vs = [], []
    with torch.no_grad():
        for x in data['x'].split(64):
            hs.append(intermediate(identity, x))
            vs.append(torch.cat((views.receiver(x), packet_time_statistics(x)), 1))
        h = torch.cat(hs)
        return dict(h=h, v=torch.cat(vs), z=identity_from_intermediate(identity, h),
                    logits=classify_intermediate(identity, h))


def packet_statistics(identity, h, y):
    z = identity_from_intermediate(identity, h)
    return dict(z=z.mean(0), normz=F.normalize(z, dim=1).mean(0),
                margin=margin_vector(classify_intermediate(identity, h), y).mean(0),
                normz_variance=F.normalize(z, dim=1).var(0, unbiased=False))


def destination_anchors(encoded, bags):
    """Fixed source-fit destination directions; only destination-role packets."""
    anchors = defaultdict(list)
    for (y, rx, day, condition), roles in bags.items():
        anchors[(rx, day, condition)].append((y, F.normalize(encoded['z'][roles[2]], dim=1).mean(0)))
    return {k: torch.stack([z for _, z in sorted(v)]).detach() for k, v in anchors.items()}


def distribution_loss(identity, moved_h, target_h, y, anchors, margin_weight=.1, anchor_weight=.1):
    pred = packet_statistics(identity, moved_h, y)
    with torch.no_grad():
        target = packet_statistics(identity, target_h, y)
    z = (pred['z'] - target['z']).square().mean()
    normz = (pred['normz'] - target['normz']).square().mean()
    margin = (pred['margin'] - target['margin']).square().mean()
    directions = F.normalize(anchors.detach(), dim=1)
    anchor = ((pred['normz'] - target['normz']) @ directions.T).square().mean()
    return z + normz + margin_weight * margin + anchor_weight * anchor, dict(
        G_mean_loss=z, normalized_G_mean_loss=normz, margin_vector_loss=margin, anchor_loss=anchor)


def _descriptor(encoded, relation, center, scale):
    # Equal-TX weighting avoids packet-count-induced transmitter imbalance.
    v0 = torch.stack([encoded['v'][a].mean(0) for a, _ in relation['donor_groups']]).mean(0)
    v1 = torch.stack([encoded['v'][b].mean(0) for _, b in relation['donor_groups']]).mean(0)
    return ((v0-center)/scale)[None], ((v1-center)/scale)[None]


def _key(r):
    return (r['y'], r['source_rx'], r['destination_rx'], r['day'], r['condition'])


def directed_mean_baselines(encoded, relations):
    """Fit-only directed RX × condition × day, leaving recipient TX out."""
    return {_key(r): torch.stack([encoded['h'][b].mean(0)-encoded['h'][a].mean(0)
        for a,b in r['donor_groups']]).mean(0).detach() for r in relations}


def receiver_audit(identity, fit, audit, generator, steps=200, *, lr=2e-4,
                   repeats=2, reference_version='frozen_source_reference_v2',
                   max_recipient_packets=32, **unused):
    """One R model jointly fitted on all conditions; independent repeated bags."""
    fmeta, ameta = validate_data(fit), validate_data(audit)
    if set(fmeta['ids']) & set(ameta['ids']):
        raise ValueError('Auxiliary fit/audit physical packet overlap')
    device = fit['x'].device
    fit_relations, fit_bags = make_relations(fit)
    audits = [make_relations(audit, repeat) for repeat in range(repeats)]
    report = dict(scope='source_L_only', target_access=False, identity_updated=False,
        joint_condition_parameters=True, conditions=sorted(set(fmeta['condition']), key=str),
        reference_version=reference_version, code_excludes_recipient_tx=True,
        condition_matching='exact source day and condition; equal TX-weight descriptors',
        bag_roles=['descriptor', 'recipient', 'destination'], independent_bags=True,
        repeat_count=repeats, fit_relations=len(fit_relations),
        audit_relations=[len(r) for r,_ in audits], audit_feedback=False,
        semantics='Source RX conditional distributions, not paired transmissions or pure hardware',
        limitations=['Same physical bags reused across repeats; repeats measure partition sensitivity, not independent samples',
            'Audit packets are held out from auxiliary fitting, not necessarily parent identity training',
            'Coordinate drift is an online integration responsibility; reference_version must change when E/G changes'])
    if not fit_relations or any(not r for r,_ in audits):
        return dict(report={**report, 'status':'INSUFFICIENT_MATCHED_GROUPS'}, state_dict={}, step_logs=[])
    with _fixed(identity):
        ef, ea = _encode(identity, fit), _encode(identity, audit)
        center = ef['v'].mean(0); scale = ef['v'].std(0, unbiased=False).clamp_min(.01)
        anchors = destination_anchors(ef, fit_bags)
        baseline = directed_mean_baselines(ef, fit_relations)
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(generator.initial_seed()+8401)
            model = ReceiverDistributionAction().to(device)
            legacy = ReceiverAction().to(device)
        opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in (model, legacy)]
        logs = []
        order = torch.randperm(len(fit_relations), generator=generator).tolist()
        for step in range(steps):
            # Cycle balanced relations; each selected bag has <= 32 recipient packets.
            r = fit_relations[order[step % len(order)]]
            if step and step % len(order) == 0:
                order = torch.randperm(len(fit_relations), generator=generator).tolist()
                r = fit_relations[order[0]]
            ix = r['recipient'][:max_recipient_packets]
            h0, h1 = ef['h'][ix], ef['h'][r['destination']]
            v0,v1 = _descriptor(ef,r,center,scale)
            opts[0].zero_grad(set_to_none=True)
            moved = h0 + model(h0,v0,v1)
            loss,parts = distribution_loss(identity,moved,h1,r['y'],anchors[(r['destination_rx'],r['day'],r['condition'])])
            loss.backward()
            grad = torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]).norm()
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(grad)):
                raise FloatingPointError('Nonfinite receiver distribution fit')
            opts[0].step()
            opts[1].zero_grad(set_to_none=True)
            c0,c1 = h0.mean(0,keepdim=True),h1.mean(0,keepdim=True)
            old = c0 + legacy(v0,v1,c0)
            oldloss = (old-c1).square().mean() + .1*(identity_from_intermediate(identity,old)-identity_from_intermediate(identity,c1)).square().mean()
            oldloss.backward();opts[1].step()
            logs.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(grad),
                legacy_centroid_loss=float(oldloss.detach()),lr=lr,condition=r['condition'],
                day=r['day'],recipient_packets=len(ix),
                **{k:float(v.detach()) for k,v in parts.items()}))
        model.eval(); legacy.eval(); records=[]
        with torch.no_grad():
            for repeat,(relations,_) in enumerate(audits):
                for r in relations:
                    if _key(r) not in baseline or (r['destination_rx'],r['day'],r['condition']) not in anchors:
                        raise ValueError('Audit source RX/day/condition has no fit-only baseline/anchor')
                    h0,h1=ea['h'][r['recipient']],ea['h'][r['destination']]
                    v0,v1=_descriptor(ea,r,center,scale)
                    targets=packet_statistics(identity,h1,r['y'])
                    source=packet_statistics(identity,h0,r['y'])
                    predictions=dict(learned=h0+model(h0,v0,v1),zero=h0,
                        directed_other_tx_mean=h0+baseline[_key(r)],
                        old_centroid=(h0.mean(0,keepdim=True)+legacy(v0,v1,h0.mean(0,keepdim=True))))
                    for mode,moved in predictions.items():
                        pred=packet_statistics(identity,moved,r['y'])
                        records.append(dict(mode=mode,repeat=repeat,y=r['y'],source_rx=r['source_rx'],
                            destination_rx=r['destination_rx'],day=r['day'],condition=r['condition'],
                            donor_tx=r['donor_tx'],recipient_count=len(h0),destination_count=len(h1),
                            G_mean_mse=float((pred['z']-targets['z']).square().mean()),
                            normalized_G_mean_mse=float((pred['normz']-targets['normz']).square().mean()),
                            margin_vector_mae=float((pred['margin']-targets['margin']).abs().mean()),
                            normalized_G_variance_mse=float((pred['normz_variance']-targets['normz_variance']).square().mean()),
                            zero_G_mean_mse=float((source['z']-targets['z']).square().mean()),
                            predicted_mean_normalized_z=pred['normz'].cpu().tolist(),
                            destination_mean_normalized_z=targets['normz'].cpu().tolist(),
                            predicted_margin_vector=pred['margin'].cpu().tolist(),
                            destination_margin_vector=targets['margin'].cpu().tolist()))
        strata=defaultdict(list)
        for record in records:strata[(record['condition'],record['mode'])].append(record)
        metrics=[dict(condition=c,mode=m,relations=len(rows),**{k:sum(r[k] for r in rows)/len(rows)
            for k in ('G_mean_mse','normalized_G_mean_mse','margin_vector_mae','normalized_G_variance_mse','zero_G_mean_mse')})
            for (c,m),rows in sorted(strata.items(),key=str)]
        repeatability=[]
        for (condition,mode),rows in sorted(strata.items(),key=str):
            per_repeat=[]
            for repeat in range(repeats):
                subset=[r for r in rows if r['repeat']==repeat]
                per_repeat.append(dict(repeat=repeat,relations=len(subset),
                    G_mean_mse=sum(r['G_mean_mse'] for r in subset)/len(subset),
                    margin_vector_mae=sum(r['margin_vector_mae'] for r in subset)/len(subset)))
            repeatability.append(dict(condition=condition,mode=mode,repeats=per_repeat,
                G_mean_mse_range=max(r['G_mean_mse'] for r in per_repeat)-min(r['G_mean_mse'] for r in per_repeat),
                margin_vector_mae_range=max(r['margin_vector_mae'] for r in per_repeat)-min(r['margin_vector_mae'] for r in per_repeat)))
        report.update(status='VERIFIED', steps=steps, optimizer_steps=2*steps,
            predictive_metrics=metrics, records=records, auxiliary_parameters=sum(p.numel() for p in model.parameters()),
            repeated_bag_stability=repeatability,
            matched_legacy_budget=True, packet_nonlinearity_before_aggregation=True,
            anchor_supervision_active=True, margin_supervision_active=True,
            source_rx_distribution={str(rx):ameta['rx'].count(rx) for rx in sorted(set(ameta['rx']))})
        state=dict(branch={k:v.detach().cpu() for k,v in model.state_dict().items()},
            legacy_branch={k:v.detach().cpu() for k,v in legacy.state_dict().items()},
            view_center=center.cpu(),view_scale=scale.cpu(),reference_version=reference_version)
    return dict(report=report,state_dict=state,step_logs=logs)
