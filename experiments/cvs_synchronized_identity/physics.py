"""Existing public TX/channel/RX cascade plus bounded affine-phase audit."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_reference_identity.physics import TX_ROWS,RX_ROWS,received


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    model.eval();p=next(model.parameters())
    x=torch.tensor(np.stack([received(t,r) for t in TX_ROWS for r in RX_ROWS]).tolist(),dtype=p.dtype,device=p.device)
    features=model.features(x);embedding=F.normalize(features,dim=1,eps=1e-4);logits=model.classify_features(features)
    omega,valid,_=model.synchronizer.estimate(x)
    audits=[]
    for theta,cfo_hz in ((.37,0.),(-1.2,80000.),(2.9,-80000.)):
        phase=x.new_tensor(theta)+2*math.pi*cfo_hz/25000000*torch.arange(256,device=p.device,dtype=p.dtype)
        r,i=x.unbind(1);y=torch.stack((r*phase.cos()-i*phase.sin(),r*phase.sin()+i*phase.cos()),1)
        other=model.features(y);logit=model.classify_features(other)
        audits.append(dict(theta_radians=theta,received_cfo_hz=cfo_hz,
            whole_logit_max_abs_error=float((logit-logits).abs().max()),
            whole_unit_embedding_max_distance=float((F.normalize(other,dim=1,eps=1e-4)-embedding).norm(dim=1).max()),
            estimated_cfo_principal_branch_crossing_count=int(((omega+cfo_hz*2*math.pi/25000000).abs()>=math.pi/20).sum()),
            valid_correlation_count=int(valid.sum())))
    records=[];isolated=[]
    for ti,tx in enumerate(TX_ROWS):
        base=ti*len(RX_ROWS)
        if ti:isolated.append(dict(tx=tx['name'],rx='identity',unit_embedding_distance_to_ideal_tx=float((embedding[base]-embedding[0]).norm())))
        for ri,rx in enumerate(RX_ROWS):
            j=base+ri
            records.append(dict(tx=tx['name'],rx=rx['name'],unit_embedding_distance_same_tx_identity_rx=float((embedding[j]-embedding[base]).norm()),source_classifier_logits=logits[j].tolist()))
    iq=float(np.max(abs(received(TX_ROWS[3],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[4]))))
    cubic=float(np.max(abs(received(TX_ROWS[1],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[5]))))
    if max(iq,cubic)>1e-12:raise ValueError('Original unidentifiable TX/RX confounds differ')
    return dict(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE',synthetic_only=True,training_augmentation=False,
        formal_data_access=False,target_access=False,model_updated=False,hardware_parameter_recovery=False,
        phase_audit=audits,phase_numeric_tolerance=1e-3,phase_tolerance_scope='Report only,not source ranking/stopping',
        phase_tolerance_pass=all(a['whole_logit_max_abs_error']<1e-3 for a in audits),
        synchronization=model.synchronizer.diagnostics(x),records=records,isolated_tx_changes=isolated,
        confounds=dict(iq_tx_rx_waveform_max_error=iq,cubic_tx_rx_waveform_max_error=cubic),
        simulated_chain='Existing100Msps steadyLSTF TX->channel->RX->12toneprojection->25Msps relativeCFO/RMS',
        exact_wisig_equalizer=False,identity_accuracy='N/A:public TX settings are not true source identity labels',
        claim='Whole constantphase and bounded affinephase behavior;no arbitraryRX/LTI or uniqueTX hardware identification')
