"""Same pure native acceptance calculation, without old experiment imports."""
import torch

FORBIDDEN_TX={'tx','tx_i','true_tx_i','tx_label','y_tx'}


def gate(native,args,teacher_logits,student_logits,extra,domain):
    metadata=native._meta_from_extra(extra)
    if any(key in metadata for key in FORBIDDEN_TX):raise ValueError('TX metadata passed to source proxy gate')
    if (args.pseudo_domain_gate or not args.pseudo_temporal_gate
        or args.pseudo_temporal_mode!='batch_neighbor' or not args.pseudo_strong_agreement):
        raise ValueError('Expected original native gate')
    confidence,pseudo=teacher_logits.softmax(1).max(1)
    threshold=native._threshold_mask(confidence,domain,args)
    temporal=native._temporal_mask_tensor(pseudo,confidence,extra,args,confidence.device)
    agreement=student_logits.argmax(1)==pseudo
    structural=native._temporal_mask_tensor(torch.zeros_like(pseudo),torch.ones_like(confidence),extra,args,confidence.device)
    accepted=threshold&temporal&agreement
    if (accepted&~structural).any():raise ValueError('Accepted mask exceeds metadata structural bound')
    return pseudo,dict(confidence=threshold,temporal=temporal,agreement=agreement,
        base=threshold&temporal,accepted=accepted,structural=structural,all=torch.ones_like(accepted))
