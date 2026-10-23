"""Use the unmodified supervised T1 graph and replace only its PL extension."""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import shutil
import time
import torch
from experiments.cric_dura_comparison.io import read, write
from .plan import validate
from .objective import Objective


@contextmanager
def installed(config, holder, *, parent_algorithm=None):
    objective_type = Objective
    if parent_algorithm is not None:
        from .view_anchor import validate as validate_view, PARENT_ALGORITHM
        if config.get('u_gradprotect'):
            from .u_gradprotect import validate as validate_u_gradprotect
            from .u_gradprotect_objective import GradientProtectedObjective
            validate_u_gradprotect(config)
            objective_type = GradientProtectedObjective
        elif config.get('u_mixstyle'):
            from .u_mixstyle import validate as validate_u_mixstyle
            from .u_mixstyle_objective import UnlabeledMixStyleObjective
            validate_u_mixstyle(config)
            objective_type = UnlabeledMixStyleObjective
        elif config.get('u_cric'):
            from .u_cric import validate as validate_u_cric
            from .u_cric_objective import UnlabeledCRICObjective
            validate_u_cric(config)
            objective_type = UnlabeledCRICObjective
        elif config.get('paired_u'):
            from .paired_u import validate as validate_pair
            from .paired_u_objective import PairedObjective
            validate_pair(config)
            objective_type = PairedObjective
        else:
            validate_view(config)
        if (parent_algorithm != PARENT_ALGORITHM or config.get('epoch_offset') != 30
                or config.get('step_offset') != 6660
                or config.get('parent_source') != str(Path(config['fork_parent_source']) / 'segment30')):
            raise ValueError('Only the declared RXPSM E30 to physical-feature transition is supported')
    from experiments.cric_dura_comparison import t1_model, f_optimizer
    original_install, original_pl, original_scope = t1_model.install, t1_model.plain_pl, f_optimizer.native_scope

    def model_install(model, c):
        model = original_install(model, c)
        objective = objective_type(model, c)
        parent = c.get('parent_source')
        if parent:
            previous = read(Path(parent) / 'complete.json')['config']
            if (previous.get('algorithm_id') != (parent_algorithm or c['algorithm_id'])
                    or any(previous.get(k) != c.get(k) for k in ('ssl', 'source_contract', 'model_seed'))):
                raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH: own SSL state')
            objective.load_state_dict(torch.load(Path(parent) / 'ssl_state.pt', map_location=next(model.parameters()).device, weights_only=False))
        model.__dict__['_pamf_ssl_objective'] = objective  # Training-only; never an inference module.
        holder['objective'], holder['model'] = objective, model
        old_hook = model._source_labeled_feature_hook

        def hook(self, **kwargs):
            step = kwargs['optimizer_step'] + c.get('step_offset', 0)
            objective.current_step, objective.current_epoch = step, (step - 1) // 222 + 1
            objective.labeled = (kwargs['labeled_inputs'], kwargs['labels'], kwargs['extra'][1]['rx_i'].to(kwargs['labels'].device))
            try:
                value, metrics = old_hook(**kwargs)
                if c.get('u_gradprotect'):
                    guided = objective.guide(kwargs['identity_loss'] + value, model)
                    value = guided - kwargs['identity_loss']
                return value, dict(metrics, **objective.metrics)
            finally:
                objective.labeled = None

        hook._accepts_labeled_inputs = True
        model._source_labeled_feature_hook = MethodType(hook, model)
        return model

    @contextmanager
    def optimizer_scope(owner, mode):
        def created(optimizer):
            optimizer.register_step_post_hook(lambda *_: owner.model._pamf_ssl_objective.after_update(owner.model))
        owner.optimizer_created = created
        with original_scope(owner, mode):
            yield

    t1_model.install = model_install
    t1_model.plain_pl = lambda model, device: model._pamf_ssl_objective.loss(model, device)
    f_optimizer.native_scope = optimizer_scope
    try:
        yield
    finally:
        t1_model.install, t1_model.plain_pl, f_optimizer.native_scope = original_install, original_pl, original_scope


def train(c, device='cuda:0'):
    validate(c)
    from experiments.cric_dura_comparison.f_train import train as segment
    root = Path(c['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root / 'resolved_config.json', c)
    parent = None
    segments = []
    start = time.perf_counter()
    for epochs, offset in ((20, 0), (10, 20), (20, 30)):
        folder = root / ('segment%02d' % (offset + epochs))
        cfg = dict(c, output_root=str(folder), segment_epochs=epochs, epoch_offset=offset,
                   step_offset=offset * 222, parent_source=parent)
        holder = {}
        with installed(cfg, holder):
            segment(cfg, device=device)
        state = holder['objective'].state_dict()
        torch.save(state, folder / 'ssl_state.pt')
        done = read(folder / 'complete.json')
        done['actual_adapter_SSL'] = dict(method=c['algorithm_id'], native_U_loop=False,
            own_EMA=True, cumulative_EMA_updates=state['ema_updates'], U_seen=state['seen'],
            physical_U_seen=state['physical_seen'], effective_U_weight=state['effective'],
            effective_U_per_class=state['per_class'].tolist(), U_truth_read=False,
            labeled_only_prototype_updates=c['ssl_method'] == 'RXPSM')
        write(folder / 'complete.json', done, exclusive=False)
        segments.append(dict(folder=str(folder), **done))
        parent = str(folder)
        write(root / 'progress.json', dict(status='SEGMENT_COMPLETE', epoch=offset + epochs,
            updates=(offset + epochs) * 222, seconds=time.perf_counter() - start,
            actual_adapter_SSL=done['actual_adapter_SSL']), exclusive=False)
    shutil.copyfile(Path(parent) / 'final_ssdg.pth', root / 'final_ssdg.pth')
    write(root / 'source_contract.json', read(c['source_contract']))
    write(root / 'complete.json', dict(status='SOURCE_FROZEN', method_id=c['algorithm_id'],
        scaffold_id=c['scaffold_id'], epoch=50, updates=11100, segments=segments,
        source_contract=c['source_contract'], config=c, initialization='same_contract_scratch20_own10_own20',
        ancestors=[s['folder'] for s in segments], actual_adapter_SSL=done['actual_adapter_SSL'],
        head_fit_updates=0, U_truth_read=False, target_access=False, seconds=time.perf_counter() - start))
