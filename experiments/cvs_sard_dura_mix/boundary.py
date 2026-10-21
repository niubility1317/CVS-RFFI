import torch

def require_training_role(role):
    if role != 'L_s':
        raise ValueError('DURA augmentation is restricted to true labeled source L_s')

def training_tensor(engine, x):
    # Inference tensors cannot be saved by F autograd. Clone after leaving G.
    with torch.inference_mode(False):
        restored = engine(x.detach())[0]
        result = restored.clone().detach()
    if result.shape != x.shape or result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise FloatingPointError('Invalid DURA output; raw fallback is forbidden')
    if torch.is_inference(result) or result.requires_grad:
        raise ValueError('Owned ordinary detached tensor required for F training')
    return result
