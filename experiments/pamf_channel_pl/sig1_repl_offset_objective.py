"""REPL candidates: remove style; optionally replace full-pair extrapolation."""
from pathlib import Path
import torch
from torch.nn import functional as F
from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.hard_pl_joint_objective import JointNoAsMixStyle,GuardedJointNoAs
from experiments.pamf_ssl_research.hard_pl_receiver_extrapolation_objective import ReceiverExtrapolation,GuardedReceiverExtrapolation
from experiments.pamf_ssl_research.hard_pl_guard_objective import Guarded
from .repl_rx_moments import LabeledReceiverMoments

METHODS=('REPL_NoStyle','REPL_NoStyleNoRXE','REPL_RXOffset')
MODES=dict(REPL_NoStyle='full_pair',REPL_NoStyleNoRXE='none',REPL_RXOffset='L_RX_offset')
GEOMETRY_RUN='20261026-diagnostic-PAMF-SIG1-REPL-RXGeometry-m1-r01'


def initial_moments(c):
    import numpy as np
    folder=Path(c['rx_initial_features']);header=read(folder/'features_complete.json')
    contract=read(c['source_contract'])
    if (header['status']!='SOURCE_SIG1_FEATURE_EXTRACTION_COMPLETE' or header['teacher_epoch']!=20 or header['model_seed']!=c['model_seed']
            or header['source_contract']!=contract or header['optimizer_updates'] or header['U_truth_read'] or header['target_reads']):
        raise ValueError('Own known frozen SIG1 E20 L-only initial moments required')
    with np.load(folder/'L_s/metadata.npz',allow_pickle=False) as data:
        if data['ids'].tolist()!=contract['role_ids']['L_s']:raise ValueError('Initial L feature physical order differs')
        y,rx=data['y'],data['rx_i']
    x=np.load(folder/'L_s/raw_features.npy',allow_pickle=False)
    if x.shape!=(6300,160) or x.dtype!=np.float32 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x,axis=1),1,atol=2e-5):
        raise ValueError('Exact normalized L6300 full features required')
    means=np.empty((6,5,160),np.float32);counts=[]
    for cls in range(6):
        row=[]
        for j,r in enumerate(c['source_rxs']):
            mask=(y==cls)&(rx==r);n=int(mask.sum());row.append(n)
            if n==0:raise ValueError('No missing initial L class/RX cell may be imputed')
            means[cls,j]=x[mask].astype(np.float64).mean(0)
        counts.append(row)
    tensor=torch.frombuffer(bytearray(means.tobytes()),dtype=torch.float32).reshape(6,5,160)
    return LabeledReceiverMoments(tensor,torch.tensor(counts,dtype=torch.long),c['source_rxs'])


class ProbeGuide:
    def guide(self,labeled,student):
        if self.c.get('candidate_smoke'):
            named=[(n,p) for n,p in student.named_parameters() if p.requires_grad]
            gradients=torch.autograd.grad(self.pending_gradient_loss,[p for _,p in named],retain_graph=True,allow_unused=True)
            norms=dict(signature=0.,head=0.,other=0.)
            for (name,_),value in zip(named,gradients):
                if value is None:continue
                if not bool(torch.isfinite(value).all()):raise ValueError('Finite actual U gradients required')
                key='signature' if 'sequence_pool.' in name else 'head' if '.cls_head.' in name else 'other'
                norms[key]+=float(value.detach().double().square().sum())
            if min(norms.values())<=0:raise ValueError('Actual U loss must train single SIG_TIME, original head and body')
            self.smoke_U_gradient_norms={k:v**.5 for k,v in norms.items()}
        return super().guide(labeled,student)


class ReceiverOffset(ReceiverExtrapolation):
    def __init__(self,c):
        super().__init__(c)
        if c.get('epoch_offset')==30:
            state=torch.load(Path(c['parent_source'])/'rx_moments.pt',map_location='cpu',weights_only=False)
            if (state['run_id']!=c['run_id'] or state['row_id']!=c['row_id'] or state['model_seed']!=c['model_seed']
                    or state['source_contract']!=read(c['source_contract']) or state['moments']['updates']!=2220):
                raise ValueError('Own E30 L statistics must resume with the matching student')
            s=state['moments'];self.moments=LabeledReceiverMoments(s['means'],s['counts'],c['source_rxs']);self.moments.load_state_dict(s)
        elif c.get('epoch_offset')==20:self.moments=initial_moments(c)
        else:raise ValueError('Only original own E20+10+20 RX-offset trajectory required')

    def loss(self,student,device):
        self._rxe_current=[];self._rxe_forward_index=0
        try:
            # Keep exactly the original joint160/shallow/middle U objectives.
            # Bypass the old full-pair extrapolation term alone.
            value=JointNoAsMixStyle.loss(self,student,device)
            if self._rxe_forward_index!=6 or len(self._rxe_current)!=2:raise ValueError('Original six U passes required')
            old=[];new=[];eligible=0
            for h,logits,pseudo,accepted,rx,partner,valid in self._rxe_current:
                z=self.moments.extrapolate(h,pseudo,accepted,rx,partner,valid)
                output=student.id_backbone.encoder.classify_features(z);mask=(accepted&valid).to(output.dtype)
                old.append((F.cross_entropy(logits,pseudo,reduction='none',label_smoothing=.01)*mask).mean())
                new.append((F.cross_entropy(output,pseudo,reduction='none',label_smoothing=.01)*mask).mean());eligible+=int(valid.sum())
            a,b=torch.stack(old).mean(),torch.stack(new).mean()
            value=value+self.settings['weight']*.25/1.25*(b-a)
            self.rxe_calls+=1;self.rxe_heads+=2;self.rxe_packets+=sum(len(v[0]) for v in self._rxe_current);self.rxe_eligible+=eligible
            self.metrics.update(HPL_loss=value.detach(),HPL_base_loss=(value/self.settings['weight']).detach(),HPL_RXE_calls=self.rxe_calls,HPL_RXE_head_calls=self.rxe_heads,
                HPL_RXE_eligible_views=self.rxe_eligible,HPL_RXE_alpha=.25,HPL_RXE_ordinary_valid_CE=a.detach(),HPL_RXE_extrapolated_valid_CE=b.detach(),
                RX_offset_mode=1.,**{'RX_'+k:v for k,v in self.moments.metrics.items()})
            return value
        finally:self._rxe_current=None

    def state_dict(self):
        state=super().state_dict()
        state['U_receiver_extrapolation'].update(mode='L_RX_offset',feature='normalized actual deployed joint160',
            direction='class-balanced current L moments minus L class-mean subspace;no U partner feature gradient',
            alpha=.25,L_moment_decay=.9,L_moment_updates=self.moments.updates,L_stat_exposures=self.moments.exposures,
            init_unique_L_records=6300,stats_updated_from_U_or_V=False)
        return state


class NoStyle(ProbeGuide,GuardedReceiverExtrapolation):pass
class NoStyleNoRXE(ProbeGuide,GuardedJointNoAs):pass
class ProtectedOffset(ProbeGuide,Guarded,ReceiverOffset):pass


def objective(c):
    return {'REPL_NoStyle':NoStyle,'REPL_NoStyleNoRXE':NoStyleNoRXE,'REPL_RXOffset':ProtectedOffset}[c['paper_method']](c)
