"""Two-forward default-head training; full source health and cost evidence."""
from collections import Counter
from contextlib import contextmanager
import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .fold import prepare
from .streams import PrivateAugment,stream
from experiments.cvs_feature_sequence50.source import train as shared_train


class Observer:
    def __init__(self,c):
        self.c=c;self.steps=0;self.coverage=Counter();self.observed=Counter();self.epochs=[];self.augmentor=None
        self.held=int(c['fold']['held_RX']);self.fit=tuple(c['fold']['fit_RX'])
        if self.held not in (1,3,4,6,8) or self.fit!=tuple(r for r in (1,3,4,6,8) if r!=self.held):
            raise ValueError('One allowed held RX and its exact four-RX complement required')
    def record(self,role,batch):
        x,y,domain,meta=batch
        if set(meta['rx_i'].tolist())-set(self.fit) or (role=='U_s' and bool((y>=0).any())):
            raise ValueError('Held RX or U TX truth entered training loader')
        self.observed[role]+=len(x)
        keys=('eq_i','rx_i','day_i','sig_i','base_index')
        values=[meta[k].tolist() for k in keys]
        self.sequence.write(json.dumps(dict(role=role,ids=list(zip(*values))),allow_nan=False)+'\n')
    def data(self,native,ctx,out):
        self.out=Path(out);self.sequence=gzip.open(self.out/'training_sequence.jsonl.gz','xt',encoding='utf-8',compresslevel=3)
        self.held_loader=prepare(native,ctx,self.record,held_rx=self.held)
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        self.roles=role_ids_from_native(ctx)
        self.held_ids=role_ids_from_native(dict(ctx,val_loader=self.held_loader))['V']
        if {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':45360,'V':21600} or len(self.held_ids)!=5400:
            raise ValueError('Actual four RX and held V roles differ')
        if set(self.held_ids)&set().union(*(set(v) for v in self.roles.values())):
            raise ValueError('Held V physical overlap')
        d.write(self.out/'source_fold.json',dict(held_RX=self.held,fit_RX=list(self.fit),effective_counts={k:len(v) for k,v in self.roles.items()},
            held_V_count=5400,held_L_used=False,source_loader_generators='named private streams; same both arms',U_truth_read=False,target_access=False))
    def source_contract(self,expected,ctx):
        if any(not set(v)<=set(expected['role_ids'][k]) for k,v in self.roles.items()) or not set(self.held_ids)<=set(expected['role_ids']['V']):
            raise ValueError('Derived source roles escape original legal carrier')
        return dict(expected,role_ids=self.roles,source_receivers=list(self.fit),held_RX=self.held,carrier_contract_ref=d.SOURCE,
            native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',held_V_ids=self.held_ids,
            checkpoint_effective_training_contract='Fresh four RX only; held V no updates; no inherited model/teacher')
    def initialize(self,m,c,out):
        self.m=m;self.initial={k:p.detach().cpu().clone() for k,p in m.named_parameters()}
        d.write(self.out/'module_files.json',{k:str(v.__file__) for k,v in sys.modules.items()
            if k in ('SSDG.train_ssdg','DataAugmentation','leo_practical.channel','leo_practical.residual')})
    @contextmanager
    def native_scope(self,native):
        original_detach=native._detach_log_mapping;original_copy=native.deepcopy;original_augment=native.make_augmentor
        def copied(value,*args,**kwargs):
            result=original_copy(value,*args,**kwargs)
            if value is getattr(self,'m',None):
                self.teacher=result;result.id_backbone.encoder._ordinary_train_packets=0;result.id_backbone.encoder._ordinary_eval_packets=0
            return result
        def augment(cfg):
            if self.augmentor is not None or not self.c['training_mechanism']['ordinary_L']:raise ValueError('One N1 ordinary augmentor only')
            with stream(20261016051,0):inner=original_augment(cfg)
            self.augmentor=PrivateAugment(inner,20261016051)
            d.write(self.out/'ordinary_base_config.json',cfg)
            return self.augmentor
        def logged(values):
            result=original_detach(values)
            if 'train/loss' in result:
                self.steps+=1;records=self.m._ordinary_latest;self.m._ordinary_latest=None
                if records is None or len(records)!=128:raise ValueError('Exactly one L128 metadata record per update required')
                self.coverage.update(records)
            return result
        native._detach_log_mapping=logged;native.deepcopy=copied;native.make_augmentor=augment
        try:yield
        finally:
            native._detach_log_mapping=original_detach;native.deepcopy=original_copy;native.make_augmentor=original_augment
            if hasattr(self,'sequence'):self.sequence.close()
    def epoch(self,m,ctx,rows,out,device):
        step=len(rows)*222
        if self.steps!=step or m._legacy_calls!=step or m._feature_l_steps!=step or m._legacy_extra_packets!=step*128 or m._legacy_actual_L_input is not None:
            raise ValueError('Same two-forward E20 update sequence required')
        if self.c['training_mechanism']['ordinary_L']!=(self.augmentor is not None):raise ValueError('Actual ordinary switch differs')
        if self.augmentor and self.augmentor._calls!=step:raise ValueError('One ordinary IQ generation per L step required')
        self.sequence.flush()
        e=m.id_backbone.encoder;params=dict(m.named_parameters());head='id_backbone.encoder.core.id_backbone.cls_head.'
        selected=[head+'weight',head+'gain','id_backbone.encoder.sequence_pool.local_projection.weight']
        if any(not params[k].requires_grad for k in selected):raise ValueError('Default head and SIG_TIME must train jointly')
        self.epochs.append(dict(epoch=len(rows),steps=step,L_forward_packets=step*256,
            actual_training_loader_packets=dict(self.observed),student_train_forward_packets=e._ordinary_train_packets,
            student_eval_forward_packets=e._ordinary_eval_packets,EMA_eval_forward_packets=self.teacher.id_backbone.encoder._ordinary_eval_packets,
            default_head_and_SIG_movement={k:float((params[k].detach().cpu()-self.initial[k]).norm()) for k in selected},
            default_fusion_gain=params[head+'gain'].detach().tanh().cpu().tolist(),
            ordinary=self.augmentor.summary() if self.augmentor else None))
        d.write(self.out/'mechanism_progress.json',dict(records=self.epochs,head_fit_updates=0,U_truth_read=False,target_access=False))
    def finish(self):
        done=d.read(self.out/'completion.json')
        if done['optimizer_steps']!=4440 or self.steps!=4440 or done['initialization']!='scratch' or done['checkpoint_sources']:
            raise ValueError('Fixed fresh E20 required')
        expected={(tx,rx,day) for tx in range(6) for rx in self.fit for day in (1,2,3)}
        if set(self.coverage)!=expected or sum(self.coverage.values())!=568320 or self.observed['L_s']!=568320:
            raise ValueError('Actual L TX/RX/day coverage incomplete')
        if self.observed['U_s']!=396368:raise ValueError('Original source-fold U exposure budget changed')
        payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if any(not torch.equal(v,self.m.state_dict()[k].detach().cpu()) for k,v in payload['model'].items()):raise ValueError('Final live/checkpoint state differs')
        reader=copy.deepcopy(self.m).eval().requires_grad_(False);device=next(reader.parameters()).device
        output=[];ce=0.
        with torch.no_grad():
            for x,y,domain,meta in self.held_loader:
                logits=reader.forward_identity_only(x.to(device))['tx_logits']
                if logits.shape!=(len(y),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite held V default-head logits required')
                pred=logits.argmax(1).cpu();truth=y.cpu()
                ce+=float(F.cross_entropy(logits,truth.to(device),reduction='sum'))
                margin=(logits.gather(1,truth.to(device)[:,None]).flatten()-logits.masked_fill(F.one_hot(truth.to(device),6).bool(),-torch.inf).max(1).values)/30
                for i in range(len(y)):
                    output.append(dict(label=int(truth[i]),prediction=int(pred[i]),day=int(meta['day_i'][i]),
                        eq=int(meta['eq_i'][i]),RX=int(meta['rx_i'][i]),sig_i=int(meta['sig_i'][i]),base_index=int(meta['base_index'][i]),margin=float(margin[i])))
        if len(output)!=5400 or any(v['RX']!=self.held for v in output):raise ValueError('Full held source V5400 required')
        cm=torch.zeros(6,6,dtype=torch.long)
        for v in output:cm[v['label'],v['prediction']]+=1
        correct=int(cm.diagonal().sum());f1=2*cm.diagonal().double()/(cm.sum(0)+cm.sum(1)).clamp_min(1)
        held=dict(count=5400,correct=correct,accuracy=correct/5400,macro_f1=float(f1.mean()),mean_CE=ce/5400,confusion=cm.tolist())
        d.write(self.out/'held_V_predictions.json',dict(role='V',count=5400,readout='FINAL defaulthead',records=output))
        last=self.epochs[-1];e=self.m.id_backbone.encoder
        result=dict(status='SOURCE_FOLD_COMPLETE',arm=self.c['arm'],held_RX=self.held,fit_RX=list(self.fit),held_V=held,seen_V=done['final_source_metrics'],
            complete_health=True,optimizer_updates=4440,epochs=20,checkpoint_sources=[],head_fit_updates=0,
            target_access=False,U_truth_read=False,L_forward_packets=1136640,training_loader_packets=dict(self.observed),
            student_U_forward_packets=e._ordinary_train_packets-1136640,
            student_source_eval_forward_packets=e._ordinary_eval_packets,initial_smoke_in_source_eval=2,
            EMA_eval_forward_packets=self.teacher.id_backbone.encoder._ordinary_eval_packets,held_V_forward_packets=5400,
            coverage=[dict(TX=k[0],RX=k[1],day=k[2],count=v) for k,v in sorted(self.coverage.items())],
            ordinary=self.augmentor.summary() if self.augmentor else None,elapsed_seconds=done['elapsed_seconds'],
            default_head_joint_training=True,actual_feature_parameters_moved=last['default_head_and_SIG_movement'])
        d.write(self.out/'result.json',result);print('SOURCE_FOLD_COMPLETE '+json.dumps(result),flush=True)
        return result


def train(c,design=None,model_api=None):
    observer=Observer(c)
    design=d if design is None else design
    view=SimpleNamespace(**{k:getattr(design,k) for k in ('validate','make_args','read','write','SOURCE','FULL_FP32_POLICY','ROOT','UPDATES')},
        source_validation=lambda model,loader,device:d.source_validation(model,loader,device,fit=observer.fit))
    shared_train(c,design=view,model_api=api if model_api is None else model_api,observer=observer)
    return observer.finish()
