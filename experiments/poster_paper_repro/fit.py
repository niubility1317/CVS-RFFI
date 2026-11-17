"""Source and support-only training for disclosed POSTER comparisons."""
import copy
import torch
import torch.nn.functional as F
from experiments.poster_paper_repro.model import PosterHomegrownAuthor, prepare_finetune


def _valid_xy(x, y):
    if x.ndim != 3 or y.ndim != 1 or x.size(0) != len(y) or len(y) < 2:
        raise ValueError('input and label shapes are incompatible')


def fit_source(model, train_x, train_y, validation_x, validation_y, *, seed,
               epochs=50, batch_size=256, patience=10, learning_rate=.001):
    _valid_xy(train_x, train_y)
    _valid_xy(validation_x, validation_y)
    if epochs <= 0 or batch_size <= 0 or patience <= 0:
        raise ValueError('invalid source training configuration')
    model = copy.deepcopy(model)
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-7)
    best, bad, best_state, history = -1., 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for indices in torch.randperm(len(train_y), generator=generator).split(batch_size):
            x, y = train_x[indices].to(device), train_y[indices].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite source loss')
            loss.backward()
            checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not checks or not bool(torch.stack(checks).all()):
                raise FloatingPointError('nonfinite source gradient')
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for start in range(0, len(validation_y), batch_size):
                y = validation_y[start:start + batch_size].to(device)
                logits = model(validation_x[start:start + batch_size].to(device))
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError('nonfinite source validation logits')
                correct += int((logits.argmax(1) == y).sum())
                count += len(y)
        val = correct / count
        history.append({'epoch': epoch, 'training_loss': total / len(train_y), 'validation_accuracy': val})
        if val > best:
            best, bad, best_state = val, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, history


def fit_support(source, support_x, support_y, *, mode, seed, epochs=10, batch_size=256, learning_rate=.001):
    _valid_xy(support_x, support_y)
    if epochs <= 0 or batch_size <= 0:
        raise ValueError('invalid support training configuration')
    torch.manual_seed(seed)
    if mode == 'legacy_project':
        model = copy.deepcopy(source)
        model.author_finetune(int(torch.unique(support_y).numel()))
    elif isinstance(source, PosterHomegrownAuthor):
        model = prepare_finetune(source, mode=mode)
    else:
        raise ValueError('requested mode requires author network')
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(support_y), generator=generator)
    validation_count = max(1, int(len(order) * .1))
    train_indices, validation_indices = order[:-validation_count], order[-validation_count:]
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=learning_rate, eps=1e-7)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for local in torch.randperm(len(train_indices), generator=generator).split(batch_size):
            indices = train_indices[local]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(support_x[indices].to(device)), support_y[indices].to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite support loss')
            loss.backward()
            checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
            if not checks or not bool(torch.stack(checks).all()):
                raise FloatingPointError('nonfinite support gradient')
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        model.eval()
        with torch.no_grad():
            logits = model(support_x[validation_indices].to(device))
            validation_accuracy = float((logits.argmax(1) == support_y[validation_indices].to(device)).float().mean())
        history.append({'epoch': epoch, 'training_loss': total / len(train_indices),
                        'validation_accuracy': validation_accuracy, 'train_count': len(train_indices),
                        'validation_count': len(validation_indices)})
    return model, history
