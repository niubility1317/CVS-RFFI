"""Scratch matched source training, fixed final20 and all held source scenes."""
import gzip,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from . import design as d,model as api
from experiments.cvs_iq_timing20.source import Observer as PriorObserver
from experiments.cvs_feature_sequence50.source import train as shared_train
from experiments.cvs_sigtime_heldcredit.held_credit import fresh_parameter_caches
from experiments.cvs_u_schedule50.counterfactual import _preserve_rng

class Observer(PriorObserver):
    def initialize(self,m,c,out):
        super().initialize(m,c,out)
        d.write(self.out/'actual_channel.json',m._joint_channel.audit())
    def epoch(self,m,ctx,rows,out,device):
        epoch=len(rows);steps=epoch*222
        if m._feature_l_steps!=steps or m._joint_calls!=steps or m._joint_L_packets!=steps*768 or m._supcon_calls!=steps or m._supcon_applied!=max(0,epoch-5)*222 or m._supcon_pending is not None or m._supcon_handles or self.optimizer._sinc_physical_clip_calls!=steps or self.optimizer._sinc_unit_steps!=steps:raise ValueError('Actual shared F/CROSS_RX/view/update budget differs')
        expected=steps*128 if self.c['arm']=='B01' else 0
        if m._joint_As_packets!=expected or sum(m._joint_coverage.values())!=expected:raise ValueError('All satellite L must be supervised, no filtering or extra draws')
        d.write(out/'satmix_progress.json',dict(epoch=epoch,steps=steps,L_packets=m._joint_L_packets,As_packets=expected,supcon_applied=m._supcon_applied,temporary_hooks_remaining=0,target_access=False))
    def scene(self,scene):
        reader=self.reader;device=next(reader.parameters()).device;labels=[];predictions=[];ce=0.;count=0
        with _preserve_rng(),torch.no_grad(),fresh_parameter_caches(reader):
            for start in range(0,1260,128):
                raw=self.held['x'][start:start+128].to(device);y=self.held['y'][start:start+128].to(device)
                ids=self.held['ids'][start:start+128];keys=('eq_i','rx_i','day_i','sig_i','base_index')
                meta={key:torch.tensor([v[i] for v in ids],dtype=torch.long) for i,key in enumerate(keys)}
                x,observed=reader._joint_channel.apply(raw,meta,namespace='source_eval/RX1/final20',scene=scene)
                z=reader.forward_identity_only(x)['tx_logits'].double()
                if z.shape!=(len(y),6) or not bool(torch.isfinite(z).all()) or len(observed)!=len(y):raise ValueError('Complete finite held source predictions required')
                ce+=float(F.cross_entropy(z,y,reduction='sum'));predictions+=z.argmax(1).cpu().tolist();labels+=y.cpu().tolist();count+=len(y)
        result=dict(count=count,accuracy=sum(a==b for a,b in zip(labels,predictions))/count,mean_CE=ce/count,macro_F1=d.macro_f1(labels,predictions),labels=labels,predictions=predictions,physical_ids=self.held['ids'],head_fit_updates=0)
        d.write(self.out/'source_readouts'/(scene+'.json'),result);return result
    def finish(self):
        done=d.read(self.out/'completion.json');payload=torch.load(self.out/'final_ssdg.pth',map_location='cpu',weights_only=False);m=self.model
        if done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch' or done['checkpoint_sources'] or payload['args']['baseline_ckpt'] or payload['args']['teacher_ckpt'] or not payload['args']['use_aug'] or payload['args'].get('source_fold_held_rx')!=1 or payload['checkpoint_selection']!='final_only' or payload['a1_ema_successful_updates']!=4440 or payload.get('ema_model') is None:raise ValueError('Full scratch20/defaulthead/originalU provenance required')
        initial=torch.load(self.out/'initial_smoke.pt',map_location='cpu',weights_only=False)['model'];W='id_backbone.encoder.core.id_backbone.cls_head.weight'
        if payload['model'][W].shape!=(6,160) or torch.equal(initial[W],payload['model'][W]):raise ValueError('Default head must train jointly')
        if any(not torch.equal(v,m.state_dict()[n].detach().cpu()) for n,v in payload['model'].items()) or len(m._joint_details)!=4440 or m._supcon_applied!=3330:raise ValueError('Frozen live state or full execution differs')
        clean=self.evaluate(payload['model']);clean['macro_F1']=d.macro_f1(clean['labels'],clean['predictions']);d.write(self.out/'source_readouts/clean.json',clean)
        # evaluate() loaded the exact final state in reader. Same named physical
        # draws apply to B00 and B01; source labels never select draws or rows.
        views={'clean':clean}
        for scene in d.SCENES:views[scene]=self.scene(scene)
        with gzip.open(self.out/'satmix_steps.jsonl.gz','xt',encoding='utf8') as f:
            for record in m._joint_details:f.write(json.dumps(record,allow_nan=False)+'\n')
        with gzip.open(self.out/'unit_steps.jsonl.gz','xt',encoding='utf8') as f:
            for record in self.optimizer._sinc_unit_lr_records:f.write(json.dumps(record,allow_nan=False)+'\n')
        return self._final_record(done,payload,views)
    def _final_record(self,done,payload,views):
        m=self.model
        coverage=[dict(TX=k[0],RX=k[1],day=k[2],scene=k[3],state=k[4],lock=k[5],count=v) for k,v in sorted(m._joint_coverage.items())]
        d.write(self.out/'satellite_supervision_coverage.json',dict(total=m._joint_As_packets,cells=coverage,all_legal_L_supervised=True,confidence_gate=False))
        native=sys.modules['SSDG.train_ssdg']
        args=type('ActualArgs',(),payload['args'])()
        ordinary=native.build_aug_base_cfg(args)
        d.write(self.out/'ordinary_augmentation_actual.json',dict(native_file=native.__file__,use_aug=payload['args']['use_aug'],base_config=ordinary,args={k:v for k,v in payload['args'].items() if k.startswith('aug_')},input_MSE_min=min(r['ordinary_raw_MSE'] for r in m._joint_details),input_MSE_max=max(r['ordinary_raw_MSE'] for r in m._joint_details),no_second_pair_redraw=True,practical_input_is_raw=True))
        def compact(v):return {k:x for k,x in v.items() if k not in ('physical_ids','labels','predictions')}
        student=m.id_backbone.encoder;teacher=self.teacher.id_backbone.encoder
        result=dict(status='SOURCE_SATMIX_COMPLETE',arm=self.c['arm'],actual_steps=4440,final_L_CE=done['final_training_L_CE'],views={k:compact(v) for k,v in views.items()},seen_V=done['final_source_metrics'],L_forward_packets=m._joint_L_packets,supcon_applied=m._supcon_applied,supcon_layer_forward_terms=sum(len(v['supcon_details']) for v in m._joint_details),As_training_observations=m._joint_As_packets,As_source_eval_observations=7560,student_train_forward_packets=student._view_train_packets,student_eval_forward_packets=student._view_eval_packets,teacher_train_forward_packets=teacher._view_train_packets,teacher_eval_forward_packets=teacher._view_eval_packets,held_forward_packets=8820,unused_fit_anchor_reads=144,smoke_forward_packets=2,elapsed_seconds=done['elapsed_seconds'],default_head_joint=True,head_fit_updates=0,target_access=False,U_truth_read=False,checkpoint_sources=[],ordinary_L=True,Ag=False,As=self.c['arm']=='B01',model='F+CROSS_RX/X3/unifiedSinc/joint160')
        d.write(self.out/'satmix_result.json',result);return result

def train(c):
    observer=Observer(c);shared_train(c,design=d,model_api=api,observer=observer);return observer.finish()
