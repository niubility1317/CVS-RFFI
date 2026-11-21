"""Limit confirmation errors with frozen, same-source E20 predictions."""
import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import torch

from experiments.cric_dura_comparison.io import read, write
from experiments.pamf_repl_weight.pseudo_observation import PseudoObservationMixin
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import GuardedReceiverExtrapolation
from . import full_suite, transfer, view_route


RUN = '20261121-phase1-pamf-tri-label01-earlyanchor-clean-m3-r01'
MODULE = 'experiments.pamf_repl_source_adapt.early_label_anchor'
SEED = view_route.SEED


def rows():
    return [dict(row_id=f'labeled_1pct-tri-e20-{name}-s{SEED}', setting='labeled_1pct', protocol_id='label01',
                 model_seed=SEED, weight=6., views=['clean'], reuses_original=False, mode=mode, floor=floor)
            for name,mode,floor in (('veto85','veto',.8),('repair85','replace',.8),('repair95','replace',.9))]


def validate(c):
    row = c.get('source_adaptation')
    folder = Path(c['output_root'])
    if folder.name in ('segment30','segment50'):
        folder = folder.parent
    if (row not in rows() or c['run_id'] != RUN or c['row_id'] != row['row_id']
            or c['protocol_id'] != 'label01' or c['model_seed'] != row['model_seed']
            or c['paper_row']['row_id'] != transfer.original_row(row)
            or c['epochs'] != 50 or c['optimizer_updates'] != 11100
            or c['hard_pl']['weight'] != 6. or c['hard_pl_gradient_guard']['weight'] != 6.
            or c['hard_pl']['threshold'] != .95
            or folder.parts[-5:] != ('runs',RUN,'F',row['row_id'],'source')
            or any(c.get(k) for k in ('target_truth','target_inputs','tests','truth','views_root','source_fold','checkpoint'))):
        raise ValueError('Only the preregistered source-only E20 anchored arms are allowed')


def configure(ctx, row, project, commit):
    c = transfer.source_configuration(ctx,row,project,commit,RUN)
    validate(c)
    return c


parent = partial(transfer.parent, validate_config=validate)


@torch.no_grad()
def anchor_revisions(current, teacher, confidence, count, accepted, floor):
    threshold = (floor + .15*count.float()/(count.float()+14.)).clamp_max(.99)
    return accepted & (teacher != current) & (confidence >= threshold)


class Anchor(GuardedReceiverExtrapolation):
    def __init__(self,c,bank):
        super().__init__(c)
        self.classes,self.confidences = bank
        self.counts = view_route.labeled_cell_counts(c).to(self.classes.device)
        self.coverage = float(self.counts[c['source_rxs']][:,c['source_days']].float().median())
        self.current = None
        self.n_revised = 0

    def validate_unlabeled_batch(self,batch):
        super().validate_unlabeled_batch(batch)
        if self.coverage < 14:
            self.current = batch[3]

    def accept_pseudo(self,student,x,confidence,pseudo):
        accepted = super().accept_pseudo(student,x,confidence,pseudo)
        if self.coverage >= 14:
            self.last_revised = 0
            return accepted
        ids,rx,day = (self.current[k].to(x.device).long() for k in ('base_index','rx_i','day_i'))
        self.current = None
        teacher = self.classes[ids]
        if bool((teacher<0).any()):
            raise ValueError('Every U exposure must match a blind E20 decision')
        prob = self.confidences[ids]
        recipe = self.c['source_adaptation']
        revise = anchor_revisions(pseudo,teacher,prob,self.counts[rx,day,teacher],accepted,recipe['floor'])
        self.last_revised = int(revise.sum())
        self.n_revised += self.last_revised
        if recipe['mode']=='veto':
            return accepted & ~revise
        pseudo.copy_(torch.where(revise,teacher,pseudo))
        confidence.copy_(torch.where(revise,prob,confidence))
        return accepted

    def loss(self,student,device):
        value=super().loss(student,device)
        self.metrics.update(HPL_E20_revised=self.last_revised,HPL_E20_revised_total=self.n_revised)
        return value

    def state_dict(self):
        result=super().state_dict()
        result['U_E20_anchor']=dict(revised=self.n_revised, mode=self.c['source_adaptation']['mode'],
                                     source_median_cell_labels=self.coverage, enabled=self.coverage<14)
        return result


class ObservedAnchor(PseudoObservationMixin,Anchor):
    pass


@torch.no_grad()
def source_teacher(c,prefix):
    from experiments.cric_dura_comparison.f_data import SourceDataset
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    from experiments.pamf_paper_completion.model import reader_scope
    contract=read(c['source_contract'])
    data=SourceDataset(c['source_cache'],'U_s',contract)
    if data.role!='U_s' or 'y' in data.metadata:
        raise ValueError('E20 anchor only predicts on TX-blind source U')
    payload=torch.load(Path(prefix['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
    with reader_scope(c['paper_row']):
        model=reader_from_payload(prefix['config'],payload,'cuda:0')
    all_ids=torch.tensor(data.metadata['base_index'].tolist(),device='cuda:0',dtype=torch.long)
    if int(all_ids.min())<0 or len(all_ids.unique())!=len(all_ids):
        raise ValueError('Expected one valid index per physical source U sample')
    slots=int(all_ids.max())+1
    classes=torch.full((slots,),-1,dtype=torch.long,device='cuda:0')
    confidences=torch.zeros(slots,device='cuda:0')
    for ids,x in zip(all_ids.split(256),data.x.split(256),strict=True):
        logits=model.classify_features(model.features(x.to('cuda:0'))).float()
        prob,label=logits.softmax(1).max(1)
        if not bool(torch.isfinite(prob).all()):
            raise ValueError('Finite E20 posteriors required')
        classes[ids],confidences[ids]=label,prob
    return classes,confidences


@contextmanager
def installed(c,holder,bank):
    from experiments.pamf_paper_completion.model import installed as paper_scope
    with paper_scope(c,holder,ssl_enabled=True,validate=validate,objective_factory=partial(ObservedAnchor,bank=bank)):
        yield
        if not c.get('hard_pl_smoke'):
            obj=holder['objective']
            obs=obj.close_pseudo_observation()
            if (obs['calls'],obs['seen'],obs['accepted']) != (obj.calls,obj.seen,obj.accepted):
                raise ValueError('Blind U observations must match the final decisions')


def prepare(rt):
    from experiments.pamf_repl_step_guard.runner import add_u_scoring
    candidates=rows()
    return add_u_scoring(rt,candidates,transfer.prepare_jobs(rt,candidates,configure))


def smoke_teachers(runtime):
    from experiments.pamf_physics_tri.full_matrix_plan import check_graph
    from experiments.pamf_physics_tri.runner import smoke
    ctx = read(runtime['context'])
    checks = []
    for seed in sorted({row['model_seed'] for row in rows()}):
        matches = [row for row in rows() if row['model_seed'] == seed]
        first = configure(ctx,matches[0],runtime['project'],runtime['code_commit'])
        bank = source_teacher(first,parent(first))
        checks.extend(smoke(runtime,rows=lambda:matches,configure=configure,check_parent=parent,
            scope=partial(installed,bank=bank),graph_check=check_graph,check_u=lambda s:s['U_E20_anchor'],
            segment_output=True,write_result=False))
        del bank
    write(Path(runtime['run_root'])/'smoke.json',dict(status='PASS',checks=checks,real_checkpoint=True,
        optimizer_updates=0,query_reads=0,U_truth_reads=0))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=('publish','dispatch','worker'))
    p.add_argument('--kind')
    for name in ('repo','output','context','ssh-config','config'):
        p.add_argument('--'+name)
    a=p.parse_args()
    if a.action=='publish':
        from .transfer_publish import publish
        publish(a.repo,a.output,a.context,a.ssh_config,run_id=RUN,module=MODULE,
                decision_run=None,gpus=full_suite.GPUS)
        return
    c=read(a.config)
    if a.action=='dispatch':
        analyze=partial(transfer.analyze, interpretation='1% E20 anchoring on historically exposed clean',source_decision_run=None)
        transfer.dispatch(c,prepare_fn=prepare,analyze_fn=analyze,module=MODULE)
    elif a.kind=='score':
        from experiments.pamf_repl_priority.runner import worker
        worker(c,'score')
    else:
        from experiments.pamf_repl_completion.runner import activate
        activate(c)
        if a.kind=='smoke':
            smoke_teachers(c)
        elif a.kind=='train':
            from experiments.pamf_physics_tri.source import train_trajectory
            prefix=parent(c)
            train_trajectory(c,prefix,partial(installed,bank=source_teacher(c,prefix)),ssl=True,physical_L=False)
        elif a.kind=='predict':
            from experiments.pamf_physics_tri.development_evaluation import predict
            predict(c)
        else:
            p.error('Unknown stage')


if __name__=='__main__':
    main()
