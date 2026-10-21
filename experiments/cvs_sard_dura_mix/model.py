from pathlib import Path
from types import MethodType
import torch
from experiments.cvs_t1_e50 import model as original
from . import design as d

classifier, source_domain_count = original.classifier, original.source_domain_count

def attach(m, c):
    d.validate(c)
    if c['restored_augmentation']['enabled']:
        from .channel import MixedPractical
        m._full_channel = MixedPractical()
    m._crossview_mode = c['restored_augmentation']['crossview_mode']
    m._mixed_original_hook = m._source_labeled_feature_hook
    m._source_labeled_feature_hook = MethodType(mixed_hook,m)
    m._feature_learning_report['DURA_augmentation'] = c['restored_augmentation']
    return m

def mixed_hook(self, **kw):
    result = self._mixed_original_hook(**kw)
    record = self._full_records[-1]
    record.update(self._full_channel.last)
    for detail in record['SupCon_details']:
        detail.update(DURA_anchors=32 if self._crossview_mode=='SAT192' else 0,
                      raw_As_anchors=32 if self._crossview_mode=='SAT192' else 0)
    return result
mixed_hook._accepts_labeled_inputs = True

def install(m, c):
    return attach(original.install(m, d.reference()), c)

def build(c, device='cpu'):
    return attach(original.build(d.reference(), device), c)

def own_model(c, device='cpu'):
    d.validate(c)
    p = Path(c['output_root'])
    done, frozen, init = [d.read(p/name) for name in ('completion.json','source_frozen.json','initialization.json')]
    if (done['config']!=c or done['epoch']!=50 or done['optimizer_steps']!=11100 or done['target_access']
            or frozen['config']!=c or frozen['selection']!='final_only' or frozen['target_access']
            or not init['scratch_only'] or init['target_contact'] or init['checkpoint_sources'] or init['ancestors']
            or d.read(p/'source_contract.json')['role_ids']!=d.read(d.SOURCE)['role_ids']):
        raise ValueError('Own source-only scratch-final provenance differs')
    cp = torch.load(p/'final_ssdg.pth', map_location='cpu', weights_only=False)
    if (cp['epoch']!=50 or cp['a1_ema_successful_updates']!=11100 or cp['run_id']!=d.RUN
            or cp['candidate_id']!=c['row_id'] or cp['args']['baseline_ckpt'] or cp['args']['teacher_ckpt']
            or cp['args']['label_epochs']!=13 or cp['args']['pseudo_epochs']!=37 or cp['checkpoint_selection']!='final_only'):
        raise ValueError('Own original default-head FINAL50 identity differs')
    m = build(c, device)
    m.load_state_dict(cp['model'], strict=True)
    return m.eval().requires_grad_(False)
