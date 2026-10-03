"""Thirty existing public TX/channel/RX examples; no training augmentation."""
import numpy as np
import torch
from torch.nn import functional as F

from experiments.cvs_reference_identity.physics import TX_ROWS, RX_ROWS, received
from experiments.cvs_equivariant_identity.model import rotate_pair
from experiments.cvs_channel_order_identity.model import _diagnostic_mode


@torch.no_grad()
def frozen_synthetic_diagnostics(model):
    p = next(model.parameters())
    x = torch.tensor(np.stack([received(t, r) for t in TX_ROWS for r in RX_ROWS]).tolist(),
                     dtype=p.dtype, device=p.device)
    with _diagnostic_mode(model, x):
        features = model.features(x)
        embedding = F.normalize(features, dim=1, eps=1e-4)
        logits = model.classify_features(features)
        audits = []
        for theta in (.37, -1.2, 2.9):
            other = model(rotate_pair(x, x.new_full((len(x),), theta)))
            audits.append(dict(theta_radians=theta, received_cfo_hz=0.,
                               whole_logit_max_abs_error=float((other - logits).abs().max())))
        measured = model.diagnostics(x)
        records = []
        for ti, tx in enumerate(TX_ROWS):
            base = ti * len(RX_ROWS)
            for ri, rx in enumerate(RX_ROWS):
                j = base + ri
                records.append(dict(tx=tx['name'], rx=rx['name'],
                    unit_embedding_distance_same_tx_identity_rx=float((embedding[j] - embedding[base]).norm()),
                    source_classifier_logits=logits[j].tolist()))
    return dict(status='FROZEN_SYNTHETIC_DIAGNOSTIC_COMPLETE', synthetic_only=True,
                formal_data_access=False, target_access=False, training_augmentation=False,
                model_updated=False, hardware_parameter_recovery=False,
                frontfilter=measured['frontfilter'], adaptive_input=measured['adaptive_input'],
                normalization=measured['normalization'], neural_residual=measured['neural_residual'],
                diagnostic_input_scopes=measured['diagnostic_input_scopes'],
                phase_audit=audits, phase_numeric_tolerance=1e-3,
                phase_tolerance_scope='Report only; not ranking or stopping',
                phase_tolerance_pass=all(a['whole_logit_max_abs_error'] < 1e-3 for a in audits),
                whole_affine_phase_invariant_claim=False, arbitrary_channel_rx_invariant_claim=False,
                dynamic_map_invertible_claim=False, records=records,
                simulated_chain='Existing100Msps steadyLSTF TX->channel->RX->12toneprojection->25Msps relativeCFO/RMS',
                exact_wisig_equalizer=False,
                identity_accuracy='N/A:public TX settings are not actual source identity labels',
                claim='Constant-phase property only; fixed-coefficient operator bound is not dynamic-map invertibility or channel recovery')
