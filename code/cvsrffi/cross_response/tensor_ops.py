"""Equal-weight complete-grid algebra; zero interaction does not imply disentanglement."""
import torch


def _grid(x):
    if x.ndim != 3 or min(x.shape) < 1 or not torch.isfinite(x).all():
        raise ValueError('expected finite complete [TX,RX,coordinate] grid')
    return x


def double_center(x):
    x = _grid(x)
    return x - x.mean(1, keepdim=True) - x.mean(0, keepdim=True) + x.mean((0, 1), keepdim=True)


def grid_decomposition(x):
    x = _grid(x)
    grand = x.mean((0, 1), keepdim=True)
    return dict(grand=grand, tx=x.mean(1, keepdim=True)-grand,
                rx=x.mean(0, keepdim=True)-grand, interaction=double_center(x))


def arithmetic_completion(x):
    """Leave-row/column-corner completion, algebraically the cross residual."""
    x = _grid(x)
    p, q, _ = x.shape
    if min(p, q) < 2:
        raise ValueError('completion requires at least 2x2')
    row, col, total = x.sum(1, keepdim=True), x.sum(0, keepdim=True), x.sum((0, 1), keepdim=True)
    return (row-x)/(q-1)+(col-x)/(p-1)-(total-row-col+x)/((p-1)*(q-1))


def identity_interaction_loss(grid):
    """Separate Ux control: mean cell squared Euclidean interaction norm."""
    if min(grid.shape[:2]) < 2:
        raise ValueError('identity interaction requires at least 2x2')
    return double_center(grid).square().sum(-1).mean()


def response_loss(prediction, target, interaction_weight=0.0):
    if prediction.shape != target.shape or interaction_weight < 0:
        raise ValueError('matched grids and nonnegative interaction weight required')
    _grid(prediction)
    target = _grid(target).detach()
    mse = (prediction-target).square().mean()
    available = min(prediction.shape[:2]) >= 2
    if interaction_weight and not available:
        raise ValueError('interaction weighting needs independent query rectangle')
    true, pred = grid_decomposition(target), grid_decomposition(prediction)
    interaction_error = (true['interaction']-pred['interaction']).square().mean() if available else None
    loss = mse + interaction_weight * interaction_error if interaction_weight else mse
    return loss, dict(total_error=mse.detach(), interaction_available=available,
                      interaction_error=None if interaction_error is None else interaction_error.detach(),
                      true_interaction=true['interaction'].detach() if available else None,
                      predicted_interaction=pred['interaction'].detach() if available else None,
                      tx_main_effect_error=(true['tx']-pred['tx']).square().mean().detach(),
                      rx_main_effect_error=(true['rx']-pred['rx']).square().mean().detach())
