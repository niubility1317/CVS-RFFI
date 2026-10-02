"""Frozen public cascade; frequency covariance is not logit invariance."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from experiments.cvs_reference_identity.physics import TX_ROWS,RX_ROWS,received
from experiments.cvs_coordinate_identity.model import rotate

@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    model.eval();p=next(model.parameters())
    x=torch.tensor(np.stack([received(t,r) for t in TX_ROWS for r in RX_ROWS]).tolist(),dtype=p.dtype,device=p.device)
    partial,omega,valid,_,alpha=model.coordinates(x)
    features=model.core.features(partial);embedding=F.normalize(features,dim=1,eps=1e-4);logits=model.classify_features(features)
    reconstruction=float((rotate(partial,alpha*omega)-x).abs().max())
    amplitude_error=float((partial.square().sum(1)-x.square().sum(1)).abs().max())
    audits=[]
    for theta,cfo_hz in ((.37,0.),(-1.2,80000.),(2.9,-80000.)):
        nu=x.new_full((len(x),),2*math.pi*cfo_hz/25000000)
        y=rotate(x,nu);r,i=y.unbind(1)
        y=torch.stack((r*math.cos(theta)-i*math.sin(theta),r*math.sin(theta)+i*math.cos(theta)),1)
        other_partial,_,other_valid,_,_=model.coordinates(y)
        expected=rotate(partial,(1-alpha)*nu);r,i=expected.unbind(1)
        expected=torch.stack((r*math.cos(theta)-i*math.sin(theta),r*math.sin(theta)+i*math.cos(theta)),1)
        crossing=(omega+nu).abs()>=math.pi/20
        eligible=valid&other_valid&~crossing
        cov_error=float((other_partial[eligible]-expected[eligible]).abs().max()) if eligible.any() else None
        other=model.core.features(other_partial);other_logits=model.classify_features(other)
        audits.append(dict(theta_radians=theta,received_cfo_hz=cfo_hz,
            whole_logit_max_abs_error=float((other_logits-logits).abs().max()),
            whole_unit_embedding_max_distance=float((F.normalize(other,dim=1,eps=1e-4)-embedding).norm(dim=1).max()),
            estimated_cfo_principal_branch_crossing_count=int(crossing.sum()),valid_correlation_count=int(valid.sum()),
            covariance_eligible_count=int(eligible.sum()),partial_waveform_covariance_max_error=cov_error))
    records=[];isolated=[]
    for ti,tx in enumerate(TX_ROWS):
        base=ti*len(RX_ROWS)
        if ti:isolated.append(dict(tx=tx['name'],rx='identity',unit_embedding_distance_to_ideal_tx=float((embedding[base]-embedding[0]).norm())))
        for ri,rx in enumerate(RX_ROWS):
            j=base+ri;records.append(dict(tx=tx['name'],rx=rx['name'],unit_embedding_distance_same_tx_identity_rx=float((embedding[j]-embedding[base]).norm()),source_classifier_logits=logits[j].tolist()))
    iq=float(np.max(abs(received(TX_ROWS[3],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[4]))))
    cubic=float(np.max(abs(received(TX_ROWS[1],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[5]))))
    if max(iq,cubic)>1e-12:raise ValueError('Original TX/RX nonidentifiability counterexamples differ')
    return dict(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE',synthetic_only=True,training_augmentation=False,
        formal_data_access=False,target_access=False,model_updated=False,hardware_parameter_recovery=False,
        normalization=model.normalization_diagnostics(x),volterra_input=model.input_diagnostics(x),alignment_strength=float(alpha),phase_audit=audits,phase_numeric_tolerance=1e-3,phase_tolerance_scope='Report only;not ranking or stopping',
        phase_tolerance_pass=all(a['whole_logit_max_abs_error']<1e-3 for a in audits if a['received_cfo_hz']==0.),
        whole_affine_phase_invariant_claim=False,coordinate_reconstruction_max_error=reconstruction,sample_amplitude_max_error=amplitude_error,
        synchronizer=model.synchronizer.diagnostics(x),records=records,isolated_tx_changes=isolated,
        confounds=dict(iq_tx_rx_waveform_max_error=iq,cubic_tx_rx_waveform_max_error=cubic),
        simulated_chain='Existing100Msps steadyLSTF TX->channel->RX->12toneprojection->25Msps relativeCFO/RMS',exact_wisig_equalizer=False,
        identity_accuracy='N/A:public TX settings are not actual source identity labels',
        claim='Whole constantphase property and partial waveform frequency covariance when valid/no branch crossing;alpha is not TX/RX oscillator fraction;no arbitraryRX/LTI invariance')
