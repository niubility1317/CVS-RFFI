"""Two student views per unique source U record; one unchanged own teacher target."""
import torch
from torch.nn import functional as F
from .objective import Objective, confidence_weight
from .paired_u import RECIPE


def paired_terms(output, teacher_features, target, weight, centers, valid):
    n = len(target)
    logits, features = output['tx_logits'], output['z_id']
    if (logits.shape != (2*n, target.shape[1]) or features.shape != (2*n, teacher_features.shape[1])
            or teacher_features.shape[0] != n or weight.shape != (n,) or valid.shape != (n,)):
        raise ValueError('Exactly aligned clean/physical U pairs required')
    target, weight = target.detach(), weight.detach()
    ce = -(torch.cat((target, target)) * logits.log_softmax(1)).sum(1)
    prototype = ce * 0.
    if centers is not None:
        anchor = (centers.detach() * target[:, :, None]).sum(1)
        prototype = (1-F.cosine_similarity(features, torch.cat((anchor, anchor)), dim=1)) * valid.repeat(2)
    distance = 1-F.cosine_similarity(features[n:], teacher_features.detach(), dim=1)
    return dict(clean_CE=RECIPE['clean_CE']*(weight*ce[:n]).mean(),
                physical_CE=RECIPE['physical_CE']*(weight*ce[n:]).mean(),
                clean_prototype=RECIPE['clean_prototype']*(weight*prototype[:n]).mean(),
                physical_prototype=RECIPE['physical_prototype']*(weight*prototype[n:]).mean(),
                physical_feature=RECIPE['physical_feature']*(weight*distance).mean())


class PairedObjective(Objective):
    def __init__(self, student, config):
        super().__init__(student, config)
        self.paired_calls = self.paired_clean_seen = self.paired_physical_seen = 0
        self.student_U_views = 0

    def loss(self, student, device):
        if self.current_epoch < RECIPE['start_epoch']:
            raise ValueError('Paired-U is defined only for the fixed E31-to-E50 continuation')
        try:
            batch = next(student._cd_U_iter)
        except StopIteration:
            student._cd_U_iter = iter(student._cd_U_loader)
            batch = next(student._cd_U_iter)
        if not bool((batch[1] == -1).all()):
            raise ValueError('U TX truth exposed')
        x, domain = batch[0].to(device), batch[2].to(device)
        meta = {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch[3].items()}
        settings = self.c['ssl']
        with torch.no_grad():
            z = self.teacher.features(x)
            p = self.teacher.classify_features(z).softmax(1)
            confidence = p.max(1).values
            decay = settings['moment_decay']
            self.mean.lerp_(confidence.mean(), 1-decay)
            self.variance.lerp_(confidence.var(unbiased=False), 1-decay)
            weight = confidence_weight(confidence, self.mean, self.variance, settings['confidence_floor'])
            if self.labeled is None:
                raise ValueError('Current visible L batch required for prototype references')
            lx, ly, lrx = self.labeled
            self.bank.update(self.teacher.features(lx.detach()), ly, lrx)
            proto, centers, valid = self.bank.distribution(z, meta['rx_i'].long(), p, settings['prototype_temperature'])
            agreement = p.argmax(1) == proto.argmax(1)
            weight *= torch.where(valid, .25+.75*agreement.float(), 1.)
            p = torch.where(valid[:, None], (1-settings['prototype_mix'])*p+settings['prototype_mix']*proto, p)
            target = p.pow(1/settings['pseudo_temperature'])
            target /= target.sum(1, keepdim=True)
            physical = student._cd_practical(x, meta, self.current_epoch, self.current_step)
        output = student.forward_identity_only(torch.cat((x, physical)), domain_labels=domain.repeat(2))
        terms = paired_terms(output, z, target, weight, centers, valid)
        loss = sum(terms.values())
        n = len(x)
        self.seen += n
        self.physical_seen += n
        self.calls += 1
        self.paired_calls += 1
        self.paired_clean_seen += n
        self.paired_physical_seen += n
        self.student_U_views += 2*n
        self.effective += float(weight.sum())
        self.per_class.scatter_add_(0, target.argmax(1), weight)
        student._cd_U_seen += n
        student._cd_U_accepted += int((weight >= .5).sum())
        self.metrics = dict(SSL_loss=loss.detach(), SSL_weight_mean=weight.mean(),
            SSL_confidence=confidence.mean(), SSL_mean=self.mean.detach(), SSL_variance=self.variance.detach(),
            SSL_ramp=1., SSL_proto_coverage=valid.float().mean(), SSL_physical_seen=self.physical_seen,
            SSL_seen=self.seen, SSL_effective=self.effective, SSL_entropy_weight=0.,
            SSL_paired_calls=self.paired_calls, SSL_paired_clean_seen=self.paired_clean_seen,
            SSL_paired_physical_seen=self.paired_physical_seen, SSL_student_U_views=self.student_U_views,
            **{'SSL_pair_'+k:v.detach() for k,v in terms.items()})
        return loss

    def state_dict(self):
        state = super().state_dict()
        state.update(paired_u=self.c['paired_u'], paired_calls=self.paired_calls,
            paired_clean_seen=self.paired_clean_seen, paired_physical_seen=self.paired_physical_seen,
            student_U_views=self.student_U_views)
        return state

    def load_state_dict(self, state):
        if state.get('paired_u') and state['paired_u'] != self.c['paired_u']:
            raise ValueError('Paired-U state belongs to another recipe')
        super().load_state_dict(state)
        for key in ('paired_calls','paired_clean_seen','paired_physical_seen'):
            setattr(self, key, state.get(key, 0))
        self.student_U_views = state.get('student_U_views', state['seen'])
