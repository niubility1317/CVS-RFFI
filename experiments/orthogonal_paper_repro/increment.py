"""Support-only incremental registration for a separate WiSig paper test."""
import torch
import torch.nn.functional as F
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.orthogonal_incremental_sei.losses import incremental_calibration_loss


def register_session(encoder, old_weights, support_x, support_y, *, epochs=50,
                     learning_rate=.08, patience=4, top_k=60, margin=.2,
                     tau_fuse=.01, lambda_align=1.6):
    if epochs <= 0 or learning_rate <= 0 or patience <= 0:
        raise ValueError('epochs, learning rate and patience must be positive')
    if support_y.ndim != 1 or support_x.size(0) != support_y.numel():
        raise ValueError('support labels must match examples')
    class_ids, class_counts = torch.unique(support_y, sorted=True, return_counts=True)
    if class_ids.numel() == 0 or not torch.all(class_counts == 5):
        raise ValueError('each new class must have exactly five support examples')
    was_training = encoder.training
    encoder.eval()
    try:
        with torch.no_grad():
            features = encoder(support_x).detach()
    finally:
        encoder.train(was_training)
    if features.ndim != 2 or old_weights.ndim != 2 or features.size(1) != old_weights.size(1):
        raise ValueError('feature and classifier dimensions must match')
    old = old_weights.detach().to(features.device).clone()
    prototypes = F.normalize(torch.stack([features[support_y == class_id].mean(0)
                                          for class_id in class_ids]), dim=1)
    new = torch.nn.Parameter(prototypes.detach().clone())
    optimizer = torch.optim.SGD([new], lr=learning_rate)
    best, bad, history = float('inf'), 0, []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = incremental_calibration_loss(features, support_y, old, new,
            new_class_ids=class_ids, prototypes=prototypes, top_k=top_k, margin=margin,
            tau_fuse=tau_fuse, lambda_align=lambda_align)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('incremental loss is nonfinite')
        loss.backward()
        if new.grad is None or not bool(torch.isfinite(new.grad).all()):
            raise FloatingPointError('incremental gradient is nonfinite')
        optimizer.step()
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best, bad = value, 0
        else:
            bad += 1
            if bad >= patience:
                break
    return torch.cat([old, new.detach()], dim=0), history
