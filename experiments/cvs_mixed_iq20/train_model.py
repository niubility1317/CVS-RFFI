"""Original SIG_TIME graph; each already generated IQ has two stochastic passes."""
from contextlib import contextmanager
from types import MethodType
import torch
from torch.nn import functional as F
from . import train_design as d
from .objective import mixed_objective,rdrop_per_sample
from . import train_channel as channel
from experiments.cvs_sigtime_rdrop_legacy import model as legacy_model

old=legacy_model.old
source_domain_count,classifier=old.source_domain_count,old.classifier


def counted_features(self,x):
    field='_mixed_train_packets' if self.training else '_mixed_eval_packets'
    setattr(self,field,getattr(self,field)+len(x))
    return self._mixed_original_features(x)


@contextmanager
def labeled_context(self):
    with self._mixed_original_labeled_scope():
        if not self.training:
            yield
            return
        if self._mixed_actual_input is not None:raise ValueError('Previous L input not consumed')
        calls=[0]
        def capture(module,args):
            if module is not self or not args or args[0].shape!=(128,2,256) or calls[0]:
                raise ValueError('Exactly one native L128 forward required')
            self._mixed_actual_input=args[0].detach().clone();calls[0]+=1
        handle=self.register_forward_pre_hook(capture)
        try:yield
        finally:handle.remove()
        if calls[0]!=1 or self._mixed_actual_input is None:raise ValueError('Actual L IQ capture failed')


def margin(logits,y):
    wrong=logits.masked_fill(F.one_hot(y,6).bool(),-torch.inf).max(1).values
    return ((logits.gather(1,y[:,None]).squeeze(1)-wrong)/30).mean().detach()


def hook(self,*,model,labeled_output,labels,extra,domains,optimizer_step,identity_loss,labeled_count):
    if (self is not model or not self.training or not torch.is_grad_enabled() or labels.shape!=(128,)
            or labels.dtype!=torch.long or not bool(((labels>=0)&(labels<6)).all())
            or labeled_count!=128 or optimizer_step!=self._feature_l_steps or optimizer_step!=self._mixed_calls+1):
        raise ValueError('Only ordered true-label L steps may invoke mixed IQ training')
    first=labeled_output['tx_logits']
    ce=F.cross_entropy(first,labels,label_smoothing=.01)
    if not torch.allclose(ce.detach(),identity_loss.detach(),atol=1e-6,rtol=1e-5):
        raise ValueError('Native CE/label smoothing/normalization differs')
    x=self._mixed_actual_input;self._mixed_actual_input=None
    if x is None or x.shape!=(128,2,256):raise ValueError('Reuse the captured actual input')
    self._mixed_calls+=1;epoch=(optimizer_step-1)//222+1
    second=self.forward_identity_only(x,domain_labels=domains)['tx_logits']
    metrics={}
    if self._mixed_config['arm']=='R_LEGACY':
        total,paired,kl=rdrop_per_sample(first,second,labels);loss=total.mean()
        metrics.update(clean_CE=paired.mean(),clean_RDrop_KL=kl.mean())
        packets=256;records=[]
    else:
        physical=channel.metadata(extra,128)
        if self._mixed_config['training_mechanism']['A_g']:
            g,gdetail=channel.generalization(x,physical,epoch)
            metrics.update(g_power_error=gdetail['max_relative_power_error'],g_waveform_MSE=gdetail['waveform_MSE'])
        else:g=x
        if self._mixed_config['training_mechanism']['A_s']:
            s,records=channel.practical(x,physical,epoch,optimizer_step)
        else:s=x;records=[]
        # IQ generation is outside both stochastic calls for each view.
        g1=self.forward_identity_only(g,domain_labels=domains)['tx_logits']
        g2=self.forward_identity_only(g,domain_labels=domains)['tx_logits']
        s1=self.forward_identity_only(s,domain_labels=domains)['tx_logits']
        s2=self.forward_identity_only(s,domain_labels=domains)['tx_logits']
        loss,parts=mixed_objective((first,second),(g1,g2),(s1,s2),labels)
        metrics.update({k:v for k,v in parts.items() if torch.is_tensor(v)})
        metrics.update(g_margin=margin(g1,labels),s_margin=margin(s1,labels))
        packets=768
        if records:
            truth=labels.detach().cpu().tolist();rx=physical['rx_i'];sat_correct=(s1.detach().argmax(1)==labels).cpu().tolist()
            for i,m in enumerate(records):m.update(TX=int(truth[i]),RX=int(rx[i]),correct=bool(sat_correct[i]))
    self._mixed_forward_packets+=packets
    self._mixed_last_records=records
    self._mixed_record_step=optimizer_step
    metrics.update(calls=self._mixed_calls,L_forward_packets=self._mixed_forward_packets,
                   clean_margin=margin(first,labels),mixed_L_loss=loss.detach(),native_CE=ce.detach(),
                   same_IQ_pairs=1.,new_U_augmentation=0.)
    # Replace native CE exactly; the native U objectives remain outside this correction.
    return loss-identity_loss,{'mixedIQ/'+k:v.detach() if torch.is_tensor(v) else v for k,v in metrics.items()}


hook._accepts_labeled_inputs=False


def attach(model,c):
    d.validate(c)
    if getattr(model,'_source_labeled_feature_hook',None) is not None:raise ValueError('Another L loss owns this model')
    model._mixed_config=c;model._mixed_calls=0;model._mixed_forward_packets=0
    model._mixed_actual_input=None;model._mixed_last_records=[];model._mixed_record_step=0
    model._mixed_original_labeled_scope=model._phase1_labeled_feature_context
    model._phase1_labeled_feature_context=MethodType(labeled_context,model)
    model._source_labeled_feature_hook=MethodType(hook,model)
    model._feature_learning_report['mixed_IQ']=c['training_mechanism']
    model._feature_learning_report['sinc_coordinates']='RAW_HZ'
    encoder=model.id_backbone.encoder
    encoder._mixed_original_features=encoder.features
    encoder._mixed_train_packets=0;encoder._mixed_eval_packets=0
    encoder.features=MethodType(counted_features,encoder)
    if type(model.get_submodule('id_backbone.encoder.core.id_backbone.sinc')).__name__=='CoordinateSinc':
        raise ValueError('Original SIG_TIME RAW_HZ required')
    return model


def install(model,c):return attach(old.install(model,d.legacy.legacy_config(c['model_seed'])),c)
def build(c,device):return attach(old.build(d.legacy.legacy_config(c['model_seed']),device),c)


def source_proof(c):
    d.validate(c);p=__import__('pathlib').Path(c['output_root'])
    done,init,freeze=(d.read(p/n) for n in ('completion.json','initialization.json','source_frozen.json'))
    if (done.get('config')!=c or done.get('epoch')!=20 or done.get('optimizer_steps')!=4440
            or done.get('logged_steps')!=4440 or done.get('target_access') or done.get('target_evaluated')
            or done.get('checkpoint_sources')!=[] or done.get('ancestors')!=[] or done.get('head_fit_updates')!=0
            or init.get('scratch_only') is not True or init.get('target_contact') or init.get('checkpoint_sources')!=[]
            or init.get('ancestors')!=[] or freeze.get('config')!=c or freeze.get('epoch')!=20
            or freeze.get('selection')!='final_only' or freeze.get('target_access')):
        raise ValueError('Own fixed-final scratchE20/default-head source proof required')
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(p/'source_contract.json'),d.read(d.SOURCE))
    m=d.read(p/'training_mechanism_complete.json')
    if (m.get('status')!='VERIFIED' or m.get('calls')!=4440
            or m.get('L_forward_packets')!=c['training_mechanism']['expected_L_forward_packets']
            or m.get('U_truth_read') or m.get('head_fit_updates') or m.get('new_U_augmentation')):
        raise ValueError('Actual L-only mixed IQ budget differs')
    actual=d.read(p/'actual_native_args.json');actual=actual.get('args',actual);expected=d.make_args(c,'cpu')
    from experiments.cvs_evidence_responsibility_train20.provenance import RECIPE_FIELDS
    if any(actual.get(k)!=getattr(expected,k) for k in RECIPE_FIELDS) or actual.get('use_aug')!=expected.use_aug:
        raise ValueError('Actual original native/source recipe differs')
    if done.get('identity_input_policy',{}).get('explicit_CFO_identity_input') is not False:
        raise ValueError('NoExplicitCFO must be active')
    return p


def own_model(c,device):
    p=source_proof(c);ck=torch.load(p/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if (ck.get('run_id')!=d.RUN or ck.get('candidate_id')!=c['row_id'] or ck.get('epoch')!=20
            or ck.get('a1_ema_successful_updates')!=4440 or ck.get('checkpoint_selection')!='final_only'
            or ck['args'].get('baseline_ckpt') or ck['args'].get('teacher_ckpt')):
        raise ValueError('Own checkpoint identity/budget differs')
    m=build(c,device);m.load_state_dict(ck['model'],strict=True)
    return m.eval().requires_grad_(False)
