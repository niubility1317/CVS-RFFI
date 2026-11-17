"""Base-class training for a disclosed, independent paper-style reproduction."""
import copy
import torch
from experiments.orthogonal_paper_repro.vectorized_loss import contrastive_loss
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.orthogonal_incremental_sei.losses import pseudo_target_cross_entropy, class_center_separation_loss
from experiments.pamf_phase2_practical.vendor_native.paper_reproduction.orthogonal_incremental_sei.pseudo_targets import make_simplex_pseudo_targets, perturb_pseudo_targets


def train_base(encoder, train_x, train_y, validation_x, validation_y, *, num_classes=60,
               num_targets=200, feature_dim=256, epochs=100, batch_size=128,
               learning_rate=.01, patience=4, seed=2026092701, callback=None):
    if epochs <= 0 or batch_size < 2 or patience <= 0 or learning_rate <= 0:
        raise ValueError('invalid optimization configuration')
    if train_x.size(0) != train_y.numel() or validation_x.size(0) != validation_y.numel():
        raise ValueError('data and label sizes differ')
    expected = list(range(num_classes))
    if torch.unique(train_y.cpu(), sorted=True).tolist() != expected:
        raise ValueError('training labels must cover every base class')
    if torch.unique(validation_y.cpu(), sorted=True).tolist() != expected:
        raise ValueError('validation labels must cover every base class')
    torch.manual_seed(seed)
    device = next(encoder.parameters()).device
    targets = make_simplex_pseudo_targets(num_targets=num_targets, feature_dim=feature_dim,
                                         total_classes=num_classes).to(device)
    perturb = perturb_pseudo_targets(targets, noise_range=.01, seed=seed)
    assigned = {i: targets[i] for i in range(num_classes)}
    optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    val_order = torch.randperm(len(validation_y), generator=generator)
    best, bad, best_epoch, best_state, history = float('inf'), 0, 0, None, []

    def objective(x, y):
        z = encoder(x)
        return (pseudo_target_cross_entropy(z, y, assigned)
                + contrastive_loss(z, y, assigned, targets, perturb)
                + class_center_separation_loss(z, y, assigned, targets))

    for epoch in range(1, epochs + 1):
        encoder.train()
        order = torch.randperm(len(train_y), generator=generator)
        total, samples = 0., 0
        for indices in order.split(batch_size):
            if indices.numel() < 2 or torch.unique(train_y[indices]).numel() < 2:
                raise ValueError('every training batch needs at least two distinct classes')
            x, y = train_x[indices].to(device), train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(x, y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('base training loss is nonfinite')
            loss.backward()
            checks = [torch.isfinite(p.grad).all() for p in encoder.parameters() if p.grad is not None]
            if not checks or not bool(torch.stack(checks).all()):
                raise FloatingPointError('base training gradient is nonfinite')
            optimizer.step()
            total += float(loss.detach()) * len(indices)
            samples += len(indices)
        encoder.eval()
        val_total, val_samples = 0., 0
        with torch.no_grad():
            for indices in val_order.split(batch_size):
                if indices.numel() < 2 or torch.unique(validation_y[indices]).numel() < 2:
                    raise ValueError('every validation batch needs at least two distinct classes')
                loss = objective(validation_x[indices].to(device), validation_y[indices].to(device))
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('base validation loss is nonfinite')
                val_total += float(loss) * len(indices)
                val_samples += len(indices)
        val_loss = val_total / val_samples
        item = {'epoch': epoch, 'training_loss': total / samples, 'validation_loss': val_loss}
        history.append(item)
        if callback is not None:
            callback(item)
        if val_loss < best:
            best, bad, best_epoch = val_loss, 0, epoch
            best_state = copy.deepcopy(encoder.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    encoder.load_state_dict(best_state)
    return ({'targets': targets.detach(), 'base_weights': targets[:num_classes].detach().clone(),
             'best_epoch': best_epoch}, history)
