"""Fixed existing source-free TX/channel/RX chain; whole-network phase audit."""
import numpy as np
import torch
from torch.nn import functional as F
from experiments.cvs_reference_identity.physics import TX_ROWS,RX_ROWS,received
from experiments.cvs_equivariant_identity.model import rotate_pair


@torch.no_grad()
def controlled_diagnostics(model):
    model.eval();parameter=next(model.parameters())
    x=torch.tensor(np.stack([received(t,r) for t in TX_ROWS for r in RX_ROWS]).tolist(),dtype=parameter.dtype,device=parameter.device)
    emb=F.normalize(model.features(x),dim=1,eps=1e-4);logits=model.classify_features(model.features(x))
    phase=[]
    for theta in (.37,-1.2,2.9):
        rotated=rotate_pair(x,x.new_tensor(theta))
        re=F.normalize(model.features(rotated),dim=1,eps=1e-4)
        phase.append(dict(theta_radians=theta,whole_logit_max_abs_error=float((model(rotated)-logits).abs().max()),
                          whole_unit_embedding_max_distance=float((re-emb).norm(dim=1).max())))
    records=[];isolated=[]
    for ti,tx in enumerate(TX_ROWS):
        base=ti*len(RX_ROWS)
        if ti:isolated.append(dict(tx=tx['name'],rx='identity',unit_embedding_distance_to_ideal_tx=float((emb[base]-emb[0]).norm())))
        for ri,rx in enumerate(RX_ROWS):
            j=base+ri
            records.append(dict(tx=tx['name'],rx=rx['name'],unit_embedding_distance_same_tx_identity_rx=float((emb[j]-emb[base]).norm()),source_classifier_logits=logits[j].tolist()))
    iq_error=float(np.max(abs(received(TX_ROWS[3],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[4]))))
    cubic_error=float(np.max(abs(received(TX_ROWS[1],RX_ROWS[0])-received(TX_ROWS[0],RX_ROWS[5]))))
    if max(iq_error,cubic_error)>1e-12:raise AssertionError('Original TX/RX confounds must coincide')
    return dict(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE',synthetic_only=True,training_augmentation=False,
        formal_data_access=False,target_access=False,model_updated=False,hardware_parameter_recovery=False,
        whole_identity_global_phase_invariance_claimed=True,phase_audit=phase,
        phase_numeric_tolerance=1e-3,phase_tolerance_scope='Report-only numerical evidence; not source selection or stopping rule',
        phase_tolerance_pass=all(r['whole_logit_max_abs_error']<1e-3 for r in phase),
        records=records,isolated_tx_changes=isolated,confounds=dict(iq_tx_rx_waveform_max_error=iq_error,cubic_tx_rx_waveform_max_error=cubic_error),
        simulated_chain='100Msps steady L-STF -> TX cubic/image/memory -> periodic FIR -> RX gain/image/cubic ->12toneprojection ->25Msps/relativeCFO/RMS',
        exact_wisig_equalizer=False,onset_transient_simulated=False,noise_simulated=False,
        identity_accuracy='N/A:synthetic settings are not actual source identities',
        claim='All identity paths respect constant complex phase; TX/RX confounds and CFO/channel sensitivity remain; no universal RX or unique TX parameter claim')
