from types import MethodType
import torch
from . import design as d, mechanism as f
from experiments.cvs_sigtime_rdrop_legacy import model as original
classifier, source_domain_count = original.classifier, original.source_domain_count


def hook(self, *, model, labeled_output, labels, extra, domains, optimizer_step, identity_loss, labeled_count):
    if (self is not model or not self.training or not torch.is_grad_enabled()
            or optimizer_step != self._feature_l_steps or optimizer_step != self._envelope_calls + 1
            or optimizer_step != self._legacy_calls + 1 or labeled_count != 128):
        raise ValueError('Only ordered actual L student context')
    x = self._legacy_actual_L_input
    self._legacy_actual_L_input = None
    if x is None or x.shape != (128, 2, 256):
        raise ValueError('Actual captured augmented L128 required')
    first = labeled_output['tx_logits']
    second = self.forward_identity_only(x, domain_labels=domains)['tx_logits']
    _, paired, native, ce1, ce2 = original.terms(first, second, labels, 1.)
    if not torch.allclose(identity_loss.detach(), ce1.detach(), atol=1e-6, rtol=1e-5):
        raise ValueError('Original smoothed CE changed')
    loss, parts = f.consistency(first, second, labels, self._envelope_arm, self._envelope_audit)
    if self._envelope_arm == 'FULL' and not torch.equal(loss.detach(), native.detach()):
        raise ValueError('FULL must equal exact original symmetric KL')
    self._envelope_calls += 1
    self._legacy_calls += 1
    self._legacy_extra_packets += 128
    self._legacy_positive_KL_calls += int(float(loss.detach()) > 0)
    metrics = {'rdrop_envelope/' + k: v for k, v in dict(parts, calls=self._envelope_calls,
        extra_L_packets=self._legacy_extra_packets, CE1=ce1.detach(), CE2=ce2.detach(),
        paired_CE=paired.detach(), actual_augmented_input_reused=1).items()}
    return .5 * (ce2 - identity_loss) + loss, metrics


hook._accepts_labeled_inputs = False


def attach(m, c):
    m._envelope_arm, m._envelope_calls, m._envelope_audit = c['arm'], 0, f.new_audit()
    m._source_labeled_feature_hook = MethodType(hook, m)
    m._feature_learning_report['rdrop_margin_envelope'] = c['training_mechanism']
    return m


def install(m, c):
    d.validate(c)
    return attach(original.install(m, d.native_config()), c)


def build(c, device):
    d.validate(c)
    return attach(original.build(d.native_config(), device), c)
