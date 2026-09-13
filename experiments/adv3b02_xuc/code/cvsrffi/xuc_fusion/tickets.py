"""Deterministic source-only tickets; capability changes order, never membership."""
from dataclasses import dataclass
from collections import Counter
import torch
from torch.utils.data._utils.collate import default_collate
from cvsrffi.cross_response.schema import SampleRecord
from cvsrffi.cross_response.sampler import CrossBlockSampler
from cvsrffi.cross_response.scheduler import FeedbackScheduler
from cvsrffi.game_tracking.data import SourceRoleDataset, labeled_source_records
from cvsrffi.game_tracking.step_context import satellite_stage
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.schedule import build_stage_state


@dataclass(frozen=True)
class Ticket:
    ticket_id: int
    epoch: int
    batch_index: int
    labeled: tuple
    unlabeled: tuple
    grid_plan: object
    weights: dict
    seed: int
    channel_seed: int
    scene: str
    probability: float
    selected: tuple

    def exposure(self, sample_ids):
        return dict(epoch=self.epoch,sample_ids=list(sample_ids),selected_mask=list(self.selected),
                    scenario=self.scene,channel_seed=self.channel_seed)


class TicketStream:
    def __init__(self, source, args, row):
        self.source,self.args,self.row = source,args,row
        self.steps = len(source.train)//args.batch_size
        if args.game_synthetic: self.steps = min(self.steps, args.game_max_steps_per_epoch or self.steps)
        self.next_epoch = 1
        self.pending = []
        self.window = None
        self.consumed = set()
        self.window_consumed = []
        self.sampler = None
        if row['carrier'] == 'v2_grid':
            records=[]
            for record in labeled_source_records(source):
                i=record['index']
                if isinstance(source.train,SourceRoleDataset):
                    raw=source.train.base.index[source.train.indices[i]]
                    identity=(raw.tx_i,raw.rx_i,raw.day_i,raw.eq_i,raw.sig_i)
                    eq=raw.eq_i
                else:
                    _,_,_,meta=source.train[i];identity=meta['sample_id'];eq=meta['eq_i']
                records.append(SampleRecord(i,record['tx'],record['rx'],record['day'],int(eq),identity))
            scheduler=FeedbackScheduler(mode='uniform',feedback_version='transaction_v2',role_policy_version='balanced_partitions_v2')
            self.sampler=CrossBlockSampler(records,P=4,Q=4,k_menu=(2,),batch_size=args.batch_size,
                seed=args.seed,steps_per_epoch=self.steps,role_policy='balanced_partitions_v2',scheduler=scheduler)
        self.boundaries={8,12,20,25,41,45,80,91,args.label_epochs+1,args.epochs+1}
        self.boundaries.update(int(v) for k,v in vars(args).items() if k.endswith('_start_epoch') and isinstance(v,int) and v>1)
        # Every baseline schedule discontinuity is bounded; continuous ramps travel with tickets.
        self.boundaries.update({int(args.aug_warmup_epochs)+1})
        if getattr(args,'xuc_daot_rc4',False):self.boundaries.update({21,41,91,161,181})

    def fill(self):
        assert not self.pending
        start=self.next_epoch
        if start>self.args.epochs: return False
        end=min(start+9,self.args.epochs,min((v-1 for v in self.boundaries if v>start),default=self.args.epochs))
        self.window=dict(start_epoch=start,end_epoch=end,ticket_ids=[],expected_scene_counts={},expected_samples=0)
        self.window_consumed=[]
        for epoch in range(start,end+1):
            # Ordinary order uses one independent generator; all methods share physical role sets.
            ordinary=torch.randperm(len(self.source.train),generator=torch.Generator().manual_seed(self.args.seed+epoch)).tolist()
            dr=getattr(self.args,'xuc_daot_rc4',False)
            uloader=self.source.unlabeled_epoch_loader(256 if dr else self.args.batch_size,epoch_index=epoch-1 if dr else max(0,epoch-self.args.label_epochs-1),
                                                      steps=self.steps,seed=self.args.seed,workers=0)
            u_batches=list(uloader.batch_sampler)
            for bi in range(1,self.steps+1):
                plan=self.sampler.next_batch() if self.sampler is not None else None
                ids=plan.indices if plan else tuple(ordinary[(bi-1)*self.args.batch_size:bi*self.args.batch_size])
                if plan is not None and (len(plan.blocks)!=4 or len(set(plan.indices))!=128):
                    raise ValueError('Formal P4Q4K2 requires four complete disjoint blocks')
                tid=(epoch-1)*self.steps+bi-1
                seed=(self.args.seed*1000003+tid*9176)%(2**31-1)
                gen=torch.Generator().manual_seed(seed)
                channel_seed=int(torch.randint(0,2**31-1,(),generator=gen))
                scenes,p=satellite_stage(epoch)
                scene=scenes[(epoch+bi-2)%len(scenes)]
                mask=tuple((torch.rand(len(ids),generator=gen)<p).tolist())
                ticket=Ticket(tid,epoch,bi,tuple(ids),tuple(u_batches[bi-1]) if dr or (epoch>self.args.label_epochs and self.args.use_unlabeled) else (),
                              plan,dict(_loss_weights(self.args,build_stage_state(epoch,self.args))),seed,channel_seed,scene,p,mask)
                self.pending.append(ticket)
        self.next_epoch=end+1
        self.window['ticket_ids']=[t.ticket_id for t in self.pending]
        self.window['expected_scene_counts']=dict(Counter({s:sum(sum(t.selected) for t in self.pending if t.scene==s) for s in {t.scene for t in self.pending}}))
        self.window['expected_samples']=sum(len(t.labeled) for t in self.pending)
        return True

    def choose(self, *, capability=False, difficulty=None):
        if not self.pending and not self.fill(): return None,False
        coverage_driven=len(self.pending)<=self.steps
        if not self.row['ticket_curriculum_enabled'] or coverage_driven:
            ticket=min(self.pending,key=lambda t:t.ticket_id)
        else:
            difficulty=difficulty or {}
            def key(t):
                score=float(difficulty.get(t.scene,0.))*sum(t.selected)/len(t.selected)
                return (-score if capability else score,t.ticket_id)
            ticket=min(self.pending,key=key)
        return ticket,coverage_driven

    def batches(self,ticket):
        batch=default_collate([self.source.train[i] for i in ticket.labeled])
        ubatch=default_collate([self.source.unlabeled[i] for i in ticket.unlabeled]) if ticket.unlabeled else None
        return batch,ubatch

    def commit(self,ticket):
        if ticket.ticket_id in self.consumed: raise ValueError('duplicate ticket consumption')
        self.pending.remove(ticket)
        self.consumed.add(ticket.ticket_id);self.window_consumed.append(ticket.ticket_id)
        if not self.pending:
            assert sorted(self.window_consumed)==sorted(self.window['ticket_ids'])
            return dict(self.window,consumed_ids=list(self.window_consumed),membership_verified=True)
        return None

    def state_dict(self):
        return dict(next_epoch=self.next_epoch,pending=self.pending,consumed=self.consumed,window=self.window,
                    window_consumed=self.window_consumed,sampler=self.sampler.state_dict() if self.sampler else None)
