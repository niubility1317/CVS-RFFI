from contextlib import contextmanager
from pathlib import Path
import copy
import torch
from torch.nn import functional as F
from . import design as d,model as api
from .mechanism import TailAverage,TAIL_EPOCHS
from experiments.cvs_rx_u_increment20.fold import prepare
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng


class Observer:
    def __init__(self,c):self.c=c;self.average=TailAverage(c['row_id']);self.tail_readouts={}
    def data(self,native,ctx,out):
        unused,self.held=prepare(native,ctx)
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        self.roles=role_ids_from_native(ctx)
        d.write(Path(out)/'source_fold.json',dict(held_RX=1,fit_RX=list(d.FIT),effective_counts={k:len(v) for k,v in self.roles.items()},
            all_training_calibration_probe_loaders_RX1_excluded=True,held_L_packets=1260,additional_L_IQ_reads=1404,
            target_access=False,U_truth_read=False,discarded_unused_anchor_packets=len(unused['y'])))
    def source_contract(self,expected,ctx):
        from cvsrffi.xuc_fusion.native import role_ids_from_native
        if role_ids_from_native(ctx)!=self.roles or {k:len(v) for k,v in self.roles.items()}!={'L_s':5040,'U_s':45360,'V':21600}:raise ValueError('Four-RX roles differ')
        if any(not set(ids)<=set(expected['role_ids'][key]) for key,ids in self.roles.items()):raise ValueError('Effective roles escape original source contract')
        return dict(expected,role_ids=self.roles,source_receivers=list(d.FIT),held_RX=1,carrier_contract_ref=d.SOURCE,
            native_role_comparison='EXACT_MATCH_CARRIER_THEN_SOURCE_FOLD',checkpoint_effective_training_contract='Four RX only; never full-five-RX initialization')
    def initialize(self,model,c,out):
        if c!=self.c:raise ValueError('One fixed trajectory required')
        self.model=model;self.reader=copy.deepcopy(model).eval();self.out=Path(out)
        self.out.joinpath('tail_states').mkdir();self.out.joinpath('source_readouts').mkdir()
    def evaluate(self,state):
        self.reader.load_state_dict(state,strict=True);self.reader.eval();device=next(self.reader.parameters()).device
        predictions=[];labels=[];ce=margin=0.;errors=0
        with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(self.reader):
            for start in range(0,1260,128):
                x=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device)
                logits=self.reader.forward_identity_only(x)['tx_logits'].double()
                if logits.shape!=(len(y),6) or not bool(torch.isfinite(logits).all()):raise ValueError('Finite default six-class logits required')
                pred=logits.argmax(1)
                other=logits.clone();other.scatter_(1,y[:,None],float('-inf'))
                ce+=float(F.cross_entropy(logits,y,reduction='sum'));errors+=int((pred!=y).sum())
                margin+=float((logits.gather(1,y[:,None]).flatten()-other.max(1).values).sum())
                predictions+=pred.cpu().tolist();labels+=y.cpu().tolist()
        return dict(count=1260,mean_CE=ce/1260,errors=errors,accuracy=1-errors/1260,mean_margin=margin/1260,
            physical_ids=self.held['ids'],labels=labels,predictions=predictions,head_fit_updates=0)
    def epoch(self,model,ctx,rows,out,device):
        epoch=len(rows)
        if model._feature_l_steps!=epoch*222:raise ValueError('Original native update budget differs')
        self.average.add(model,epoch,trajectory_id=self.c['row_id'])
        if epoch in TAIL_EPOCHS:
            snapshot={name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
            torch.save(dict(model=snapshot,epoch=epoch,trajectory_id=self.c['row_id'],sinc_coordinates='CENTER_25K'),self.out/'tail_states'/('epoch'+str(epoch)+'.pth'))
            self.tail_readouts[epoch]=self.evaluate(snapshot)
            d.write(self.out/'source_readouts'/('epoch'+str(epoch)+'.json'),self.tail_readouts[epoch])
        d.write(self.out/'dropout_progress.json',dict(status='RUNNING',epoch=epoch,actual_steps=model._feature_l_steps,
            mode=self.c['arm'],tail_epochs=list(self.average.epochs),target_access=False,U_truth_read=False,head_fit_updates=0))
    @contextmanager
    def native_scope(self,native):yield
    def finish(self):
        done=d.read(self.out/'completion.json')
        if done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources']:raise ValueError('Full fresh trajectory required')
        payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False)
        if payload.get('ema_model') is None:raise ValueError('Original own EMA must be retained')
        tail_final=torch.load(self.out/'tail_states'/'epoch20.pth',map_location='cpu',weights_only=False)['model']
        if set(tail_final)!=set(payload['model']) or any(not torch.equal(value,tail_final[name]) for name,value in payload['model'].items()):
            raise ValueError('Cached epoch20 default-head predictions must match the final checkpoint')
        average=self.average.state_dict()
        for name,value in payload['model'].items():
            if not torch.equal(value,self.model.state_dict()[name].detach().cpu()):raise ValueError('Final live model/checkpoint differ')
        states=dict(EMA=payload['ema_model'],TAIL_AVG=average)
        exports=dict(FINAL=self.tail_readouts[20])
        for name,state in states.items():
            torch.save(dict(model=state,epoch=20,initialization='own_same_trajectory',source_run_id=d.RUN,source_row_id=self.c['row_id'],
                export=name,sinc_coordinates='CENTER_25K',default_head_included=True,tail_epochs=list(TAIL_EPOCHS) if name=='TAIL_AVG' else None),self.out/(name.lower()+'.pth'))
            exports[name]=self.evaluate(state)
            d.write(self.out/'source_readouts'/(name+'.json'),exports[name])
        final=exports['FINAL'];d.write(self.out/'source_readouts'/'FINAL.json',final)
        def paired(candidate):
            before=final['predictions'];after=candidate['predictions'];labels=final['labels']
            return dict(CE_difference=candidate['mean_CE']-final['mean_CE'],error_difference=candidate['errors']-final['errors'],
                repaired=sum(b!=y and a==y for b,a,y in zip(before,after,labels)),broken=sum(b==y and a!=y for b,a,y in zip(before,after,labels)))
        def compact(value):return {key:v for key,v in value.items() if key not in ('physical_ids','labels','predictions')}
        result=dict(status='SOURCE_DROPOUT_COMPLETE',mode=self.c['arm'],epoch=20,actual_steps=4440,source_health=done['final_training_L_CE']<=.5,
            final_training_L_CE=done['final_training_L_CE'],exports={key:compact(value) for key,value in exports.items()},
            paired_exports={key:paired(exports[key]) for key in ('EMA','TAIL_AVG')},
            tail_prediction_change_counts={str(epoch):sum(a!=b for a,b in zip(value['predictions'],final['predictions'])) for epoch,value in self.tail_readouts.items()},
            tail_epochs=list(self.average.epochs),averaged_parameter_tensors=len(self.average.sums),constant_buffers=len(self.average.buffers),
            held_L_forward_packets=7560,smoke_forward_packets=2,additional_L_IQ_reads=1404,actual_extra_optimizer_steps=0,
            target_access=False,U_truth_read=False,head_fit_updates=0,default_head_joint_training=True)
        d.write(self.out/'dropout_result.json',result);print('DROPOUT_RESULT '+str({k:v for k,v in result.items() if k in ('mode','exports','paired_exports')}),flush=True)
        return result


def train(c):
    observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer)
    return observer.finish()
