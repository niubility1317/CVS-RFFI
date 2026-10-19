import torch
from . import design as d
from .models import ActionRestorer
from experiments.pamf_residual_restore.runtime import load_coarse, parent_metadata
from experiments.pamf_reception_restore.models import ComplexTFUNet


def save_checkpoint(path, model, optimizer, epoch, updates, parent):
    torch.save(dict(schema='learned_action_checkpoint_v1', config=d.configuration(), model=model.state_dict(),
        optimizer=optimizer.state_dict(), epoch=epoch, updates=updates, parent=parent,
        training_TX_labels_read=0, training_F_access=False, target_training=False), path)


def validate_payload(payload):
    if (payload['schema'] != 'learned_action_checkpoint_v1' or payload['config'] != d.configuration() or
            payload['training_TX_labels_read'] != 0 or payload['training_F_access'] or payload['target_training']):
        raise ValueError('Action checkpoint config/training permissions differ')
    from experiments.pamf_residual_restore import design as parent_design
    parent = payload['parent']
    if (parent['run'] != parent_design.PARENT_RUN or parent['commit'] != parent_design.PARENT_COMMIT or
            parent['candidate'] != parent_design.COARSE_NAME or parent['selector'] != 'source_waveform_A' or
            parent['training_TX_labels_read'] != 0 or parent['target_training']):
        raise ValueError('Action parent lineage differs')
    if not 0 <= payload['updates'] <= 4940 or not 1 <= payload['epoch'] <= 20:
        raise ValueError('Action checkpoint budget differs')


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    validate_payload(payload)
    model = ActionRestorer(ComplexTFUNet()).to(device)
    model.load_state_dict(payload['model'], strict=True)
    return model.eval(), payload
