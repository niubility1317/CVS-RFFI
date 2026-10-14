import gzip
import json
from contextlib import contextmanager
import torch
from . import design as d
from .objective import group_risk
from experiments.cvs_local_order50.observer import derivative
from experiments.cvs_source_layer_order.probe import gradient_stats


class Observer:
    def __init__(self, *, design=None):
        self.design = design or d

    def initialize(self, model, c, out):
        d = self.design
        d.validate(c)
        self.c, self.model, self.out = c, model, out
        self.steps = self.U_steps = self.U_packets = self.U_accepted = 0
        self.logq = torch.zeros(5, dtype=torch.float64)
        named = dict(model.named_parameters())
        self.params = [named[n] for n in d.PARAMETERS]
        self.head = model.id_backbone.encoder.core.id_backbone.cls_head.weight
        if not all(p.requires_grad for p in self.params + [self.head]):
            raise ValueError('Default feature/head joint learning disabled')
        self.saved, self.grad, self.records = [], [], []
        self.stream = gzip.open(out / 'receiver_risk_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3)

    @contextmanager
    def native_scope(self, n):
        if n._labeled_identity_loss_hook is not None or n._trusted_pseudo_loss_hook is not None:
            raise ValueError('Another supervised/U loss owner active')
        n._labeled_identity_loss_hook, n._trusted_pseudo_loss_hook = self.labeled, self.unlabeled
        try:
            yield
        finally:
            n._labeled_identity_loss_hook = n._trusted_pseudo_loss_hook = None
            self.stream.close()

    def labeled(self, *, model, args, epoch, batch_idx, logits, labels, receivers,
                clean_count, original_loss, role):
        d = self.design
        step = (epoch - 1) * 222 + batch_idx
        if model is not self.model or step != self.steps + 1 or role != 'L_s':
            raise ValueError('Actual labelled risk step differs')
        if clean_count != len(labels) or len(logits) != clean_count or args.label_smoothing != .01:
            raise ValueError('Receiver risk requires single original labelled view')
        mode = self.c['receiver_risk']['mode']
        selected, updated, detail = group_risk(logits, labels, receivers, self.logq, mode, role=role)
        if abs(detail['original_CE'] - float(original_loss.detach())) > 2e-6:
            raise ValueError('Original supervised objective is not the registered CE')
        if mode == 'ERM':
            selected = original_loss  # Preserve exact original graph and value.
        self.logq = updated
        self.steps = step
        if epoch in d.DIAGNOSTIC_EPOCHS and batch_idx % 16 == 0:
            delta = selected - original_loss
            objects = self.params + [self.head]
            vectors = {key: derivative(loss, objects) for key, loss in
                       [('original', original_loss), ('selected', selected), ('delta', delta)]}
            self.grad.append(dict(step=step, epoch=epoch, vectors=vectors))
            self.records.append(dict(step=step, epoch=epoch,
                layers=[dict(parameter=name, selected=gradient_stats(vectors['original'][k], vectors['selected'][k]),
                             delta=gradient_stats(vectors['original'][k], vectors['delta'][k]))
                        for k, name in enumerate([*d.PARAMETERS, 'existing_default_W'])]))
        self.stream.write(json.dumps(dict(step=step, epoch=epoch, batch=batch_idx, mode=mode,
            source_packets=clean_count, **detail), allow_nan=False) + chr(10))
        return selected

    def unlabeled(self, *, model, args, epoch, batch_idx, loss_l, loss_u, loss_ent,
                  active, logits, pseudo, mask):
        if model is not self.model or self.steps != (epoch - 1) * 222 + batch_idx:
            raise ValueError('Original U hook is not paired to labelled step')
        if bool(active) != (epoch >= 34) or (args.lambda_u, args.lambda_ent, args.ema_decay) != (.16, .01, .999):
            raise ValueError('Original oldgate/EMA U objective differs')
        if active:
            self.U_steps += 1; self.U_packets += len(pseudo); self.U_accepted += int(mask.sum())
        return loss_u

    def epoch(self, model, ctx, rows, out, device):
        d = self.design
        epoch = int(rows[-1]['epoch']); self.stream.flush()
        if self.steps != epoch * 222:
            raise ValueError('Receiver risk source budget incomplete')
        if epoch in d.DIAGNOSTIC_EPOCHS:
            if len(self.grad) != 13:
                raise ValueError('Actual risk gradient diagnostic incomplete')
            path = out / f'receiver_risk_gradients_e{epoch}.pt'
            torch.save(self.grad, path)
            actual = torch.load(path, map_location='cpu', weights_only=False)
            ref = [r for r in self.records if r['epoch'] == epoch]
            for a, b in zip(actual, ref):
                for k, row in enumerate(b['layers']):
                    for key in ('selected', 'delta'):
                        if gradient_stats(a['vectors']['original'][k], a['vectors'][key][k]) != row[key]:
                            raise ValueError('Saved receiver-risk gradient recount differs')
            self.saved.append(epoch); self.grad = []
            d.write(out / 'receiver_risk_gradient_records.json', dict(status='VERIFIED',
                epochs=self.saved, batch_records=self.records, saved_gradient_recount='VERIFIED',
                head_fit_updates=0, default_W_normal_joint_CE=True, U_truth_read=False))

    def finish(self):
        d = self.design
        if self.steps != 11100 or self.U_steps != 3774 or self.U_packets != 963900:
            raise ValueError('Receiver risk/native U exposure incomplete')
        if self.saved != list(d.DIAGNOSTIC_EPOCHS) or len(self.records) != 52:
            raise ValueError('Receiver risk gradient epochs incomplete')
        value = dict(status='VERIFIED', mode=self.c['receiver_risk']['mode'], steps=self.steps,
            U_steps=self.U_steps, U_packets=self.U_packets, U_accepted=self.U_accepted,
            gradient_batches=52, layer_records=208, saved_vectors=624, gradient_epochs=self.saved,
            extra_model_forwards=0, new_parameters=0, new_heads=0, head_fit_updates=0,
            default_head_jointly_trained=True, U_loss_returned_unchanged=True, U_truth_read=False,
            final_group_weights=self.logq.softmax(0).tolist(), saved_gradient_recount='VERIFIED')
        d.write(self.out / 'receiver_risk_training_observation.json', value)
        return value
