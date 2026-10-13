"""Freeze three Phase1 readouts from legal source L before clean prediction.

The original baseline decisions are reused. This module never opens truth and
never imports the independent scorer. These source-fitted readouts are not Phase2.
"""
import json
import math
from pathlib import Path
import time
from . import design as d

COUNT = d.QUERY_COUNT


def validate_episode_counts(c, value):
    expected = dict(status='VERIFIED', calls=d.UPDATES, episodes=d.EPISODES,
                    support_packets=d.EPISODES * 12, query_packets=d.EPISODES * 24,
                    query_RX_episodes={str(rx): d.EPISODES // len(d.SOURCE_RX) for rx in d.SOURCE_RX},
                    U_opened=False, U_truth_access=False, V_training=False, target_access=False,
                    persistent_episode_W=False,
                    gradient_object='joint160 support and query; class weight excluded')
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Episodic source budget or input role differs')
    if type(value.get('unique_L_visited')) is not int or not 0 < value['unique_L_visited'] <= 6300:
        raise ValueError('Episodic L physical coverage count differs')
    return value


def source_proof(c):
    d.validate(c)
    if c['arm'] == 'LEGACY50':
        from experiments.cvs_frequency_context50.predict import source_proof as parent_proof
        return parent_proof(c['source_config'])
    p = Path(c['output_root'])
    done = d.read(p / 'completion.json')
    resolved = d.read(p / 'resolved_config.json')
    init = d.read(p / 'initialization.json')
    frozen = d.read(p / 'source_frozen.json')
    args = d.read(p / 'resolved_native_args.json')
    if (done.get('status') != 'SOURCE_TRAINED' or done.get('config') != c
            or done.get('epoch') != c['epochs'] or done.get('logged_steps') != c['updates']
            or done.get('optimizer_steps') != c['updates'] or done.get('target_access') is not False
            or done.get('target_evaluated') is not False or done.get('checkpoint_sources') != []):
        raise ValueError('Source budget or role differs')
    if (init.get('scratch_only') is not True or init.get('checkpoint_sources') != []
            or init.get('ancestors') != [] or init.get('target_contact') is not False
            or init.get('seed') != c['model_seed'] or init.get('source_roles') != 'EXACT_MATCH'):
        raise ValueError('Scratch provenance differs')
    actual = dict(c['source_episode'], class_head='unchanged cosine6x160', pseudo_EMA='historical unmodified')
    if (any(resolved.get(k) != v for k, v in c.items())
            or resolved.get('identity_input_policy', {}).get('explicit_CFO_identity_input') is not False
            or resolved.get('actual_source_episode') != actual):
        raise ValueError('Resolved episodic feature recipe differs')
    if (args.get('from_scratch') is not True or args.get('baseline_ckpt') != ''
            or args.get('teacher_ckpt') != '' or args.get('label_epochs') != d.LABEL_EPOCHS
            or args.get('pseudo_epochs') != d.PSEUDO_EPOCHS or args.get('epochs') != d.EPOCHS
            or any(args.get(k) != v for k, v in d.LEGACY_FIELDS.items())):
        raise ValueError('Historical pseudo/EMA/cos execution differs')
    expected_freeze = dict(status='SOURCE_FROZEN', config=c, epoch=c['epochs'], updates=c['updates'],
                           checkpoint=str(p / 'final_ssdg.pth'), selection='final_only', target_access=False)
    if frozen != expected_freeze:
        raise ValueError('Final source freeze differs')
    validate_episode_counts(c, d.read(p / 'episode_counts.json'))
    from experiments.cvs_phase1_stack.recover import verify_contract
    verify_contract(d.read(p / 'source_contract.json'), d.read(d.SOURCE))
    return p


def own_model(c, device):
    import torch
    from types import SimpleNamespace
    d.validate(c)
    if c['arm'] == 'LEGACY50':
        from experiments.cvs_frequency_context50.predict import own_model as parent_model
        return parent_model(c['source_config'], device)
    p = source_proof(c)
    from experiments.cvs_phase1_overlay.model import native_modules
    native_modules()  # Resolve native pickled argument classes before loading.
    checkpoint = torch.load(p / 'final_ssdg.pth', map_location='cpu', weights_only=False)
    args = checkpoint['args']
    if (checkpoint.get('epoch') != c['epochs'] or checkpoint.get('a1_ema_successful_updates') != c['updates']
            or checkpoint.get('run_id') != d.RUN or checkpoint.get('candidate_id') != c['row_id']
            or checkpoint.get('checkpoint_selection') != 'final_only'
            or args.get('baseline_ckpt') != '' or args.get('teacher_ckpt') != ''
            or args.get('from_scratch') is not True):
        raise ValueError('Own final episodic checkpoint differs')
    from .model import installed
    with installed(c, training=False) as native:
        if getattr(native, '_source_feature_auxiliary_hook', None) is not None:
            raise ValueError('Prediction must not install source-episode training hooks')
        model = native.build_baseline_model(SimpleNamespace(**checkpoint['baseline_args']), torch.device(device))
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.eval().requires_grad_(False)


def baseline_prediction_root(c):
    return Path(d.PROJECT) / 'runs' / d.BASELINE_EVAL_RUN / c['row_id'] / 'prediction'


def baseline_original(c, ids):
    import numpy as np
    if c['arm'] != 'LEGACY50':
        raise ValueError('Only the frozen baseline can reuse old original-head predictions')
    p = baseline_prediction_root(c)
    done = d.read(p / 'complete.json')
    source = c['source_config']
    if (done.get('status') != 'PREDICTIONS_COMPLETE' or done.get('config') != source
            or done.get('count') != COUNT or done.get('views') != list(d.VIEWS)
            or done.get('truth_read') is not False or done.get('query_fit') is not False):
        raise ValueError('Baseline original predictions are incomplete or mismatched')
    expected = dict(status='VERIFIED', config=source, checkpoint=str(Path(source['output_root']) / 'final_ssdg.pth'),
                    own_final50=True, ancestors=[], truth_read=False, query_fit=False, source_roles='EXACT_MATCH')
    if d.read(p / 'provenance.json') != expected:
        raise ValueError('Baseline original prediction provenance differs')
    with np.load(p / 'predictions.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'ids', 'clean'} or not np.array_equal(saved['ids'], ids):
            raise ValueError('Baseline original prediction IDs differ')
        value = saved['clean'].copy()
    validate_decisions(value)
    return value


def validate_decisions(value):
    if (value.shape != (COUNT,) or value.dtype.kind not in 'iu'
            or value.min() < 0 or value.max() >= len(d.CLASSES)):
        raise ValueError('Invalid fixed six-class decisions')


def checkpoint_path(c):
    source = c['source_config'] if c['arm'] == 'LEGACY50' else c
    return str(Path(source['output_root']) / 'final_ssdg.pth')


def validate_head_freeze(c, value):
    expected = dict(status='SOURCE_READOUTS_FROZEN', config=c, readouts=list(d.READOUTS),
                    feature_schema='actual joint160; row-wise L2 eps1e-4', feature_dimension=160,
                    old_W_discarded=True, feature_updates=0, feature_parameters_buffers_unchanged=True,
                    fit_L_count=6300, fit_classes=list(range(6)), fit_source_roles='EXACT_MATCH',
                    fit_U_opened=False, fit_V_training=False, target_fit=False, truth_read=False,
                    checkpoint=checkpoint_path(c), seed=c['model_seed'] + 60013,
                    source_fit=dict(c['fresh_cosine_probe']))
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError('Readouts were not frozen with the fixed legal source recipe')
    audit = value.get('fresh_cosine_audit', {})
    expected_updates = math.ceil(value['fit_L_count'] / c['fresh_cosine_probe']['batch_size']) * c['fresh_cosine_probe']['epochs']
    if (audit.get('epochs') != c['fresh_cosine_probe']['epochs'] or audit.get('updates') != expected_updates
            or audit.get('feature_updates') != 0 or len(audit.get('losses', [])) != audit['epochs']
            or not all(math.isfinite(v) for v in audit['losses'])):
        raise ValueError('Fresh cosine fitting budget differs')
    shapes = value.get('feature_parameter_shapes', {})
    if (not shapes or any(name.endswith('cls_head.weight') for name in shapes)
            or any(shape == [6, 160] for shape in shapes.values())
            or sum(math.prod(shape) for shape in shapes.values()) != value.get('feature_parameters')):
        raise ValueError('Class-independent feature parameter inventory differs')
    if value.get('source_feature_max_difference') != 0.:
        raise ValueError('Source feature boundary changed after discarding the old classifier')
    return value


def predict(row):
    import numpy as np
    import torch
    from experiments.cvs_equivariant_identity.precision import numerical_context
    from experiments.cvs_report_ssl.evaluate import iq_tensor
    from experiments.cvs_source_transfer_probe.run import data, extract
    from experiments.cvs_source_transfer_probe.readout import fit, logits
    from experiments.cvs_transfer_features.model import from_phase1
    c = d.config(next(r for r in d.rows() if r['row_id'] == row))
    source = source_proof(c)
    output = d.BASE / row / 'prediction'
    if output.exists():
        raise FileExistsError('Existing prediction must be reconciled, never overwritten')
    torch.set_num_threads(2)
    started = time.perf_counter()
    device = 'cuda:0'
    with numerical_context(d.FULL_FP32_POLICY):
        model = own_model(c, device)
        features = from_phase1(model).to(device).eval().requires_grad_(False)
        before = {k: v.clone() for k, v in features.state_dict().items()}
        with torch.no_grad():
            smoke = features.forward_features(torch.zeros(2, 2, 256, device=device))
        if smoke.shape != (2, 160) or not torch.isfinite(smoke).all():
            raise ValueError('Own final checkpoint no-query feature smoke failed')
        L, V, contract = data()
        del V
        from experiments.cvs_phase1_stack.recover import verify_contract
        verify_contract(contract, d.read(d.SOURCE))
        if (len(L['ids']) != 6300 or len(set(L['ids'])) != 6300
                or set(L['ids']) != set(contract['role_ids']['L_s'])):
            raise ValueError('Readout fit must use exact legal L_s physical samples')
        output.mkdir(parents=True, exist_ok=False)
        fit_started = time.perf_counter()
        z, unused, max_difference = extract(model, features, L, device)
        del unused
        y = L['y'].long()
        weight, audit = fit(z, y, c['model_seed'] + 60013,
                            epochs=c['fresh_cosine_probe']['epochs'], batch_size=c['fresh_cosine_probe']['batch_size'])
        prototype = torch.stack([z[y == cls].mean(0) for cls in range(6)])
        if any(v.shape != (6, 160) or not torch.isfinite(v).all() or v.requires_grad for v in (weight, prototype)):
            raise ValueError('Frozen source classifiers are malformed')
        if any(not torch.equal(v, features.state_dict()[k]) for k, v in before.items()):
            raise ValueError('Source fitting updated the frozen feature extractor')
        inventory = {name: list(p.shape) for name, p in features.named_parameters()}
        frozen = dict(status='SOURCE_READOUTS_FROZEN', config=c, readouts=list(d.READOUTS),
                      feature_schema='actual joint160; row-wise L2 eps1e-4', feature_dimension=160,
                      old_W_discarded=True, feature_updates=0, feature_parameters_buffers_unchanged=True,
                      fit_L_count=len(y), fit_classes=sorted(y.unique().tolist()), fit_source_roles='EXACT_MATCH',
                      fit_U_opened=False, fit_V_training=False, target_fit=False, truth_read=False,
                      checkpoint=checkpoint_path(c), seed=c['model_seed'] + 60013,
                      source_fit=dict(c['fresh_cosine_probe']), fresh_cosine_audit=audit,
                      feature_parameters=sum(p.numel() for p in features.parameters()),
                      feature_parameter_shapes=inventory, source_feature_max_difference=max_difference,
                      source_fit_seconds=time.perf_counter() - fit_started)
        validate_head_freeze(c, frozen)
        torch.save(dict(row_id=row, config=c, fresh_cosine=weight, prototype=prototype,
                        feature_updates=0, target_fit=False), output / 'source_heads.pth')
        d.write(output / 'head_frozen.json', frozen)
        proof = dict(status='VERIFIED', config=c, checkpoint=checkpoint_path(c), source_root=str(source),
                     source_roles='EXACT_MATCH', ancestors=[], own_final50=True,
                     old_W_discarded_for_new_readouts=True, heads_frozen_before_target_IQ=True,
                     readouts=list(d.READOUTS), truth_read=False, query_fit=False,
                     phase1_source_readouts=True, Phase2_claim=False,
                     baseline_original_reused=c['arm'] == 'LEGACY50',
                     baseline_prediction_root=str(baseline_prediction_root(c)) if c['arm'] == 'LEGACY50' else None)
        d.write(output / 'provenance.json', proof)
        # All source fitting and feature/head freezing above precede target IQ.
        from experiments.cvs_frequency_context50.predict import observation_metadata
        ids, observations = observation_metadata()
        values = {name: np.empty(COUNT, dtype=np.int64) for name in d.READOUTS}
        if c['arm'] == 'LEGACY50':
            values['ORIGINAL_HEAD'] = baseline_original(c, ids)
        array = np.load(observations['clean_ref'], mmap_mode='r', allow_pickle=False)
        if array.shape != (COUNT, 2, 256) or array.dtype.name != 'float32':
            raise ValueError('Fixed clean observation array differs')
        weight, prototype = weight.to(device), prototype.to(device)
        head = model.id_backbone.encoder.core.id_backbone.cls_head
        infer_started = time.perf_counter()
        with torch.no_grad():
            for begin in range(0, COUNT, 256):
                x = iq_tensor(np.ascontiguousarray(array[begin:begin + 256]), device)
                joint = features.forward_features(x)
                q = torch.nn.functional.normalize(joint, dim=1, eps=1e-4)
                scores = dict(FRESH_COSINE_ALL_L=logits(q, weight), COSINE_PROTO_ALL_L=logits(q, prototype))
                if c['arm'] != 'LEGACY50':
                    scores['ORIGINAL_HEAD'] = head.classify(joint)
                for name, score in scores.items():
                    if score.shape != (len(x), 6) or not torch.isfinite(score).all():
                        raise FloatingPointError('Nonfinite fixed Phase1 readout: ' + name)
                    values[name][begin:begin + len(x)] = score.argmax(1).cpu().tolist()
        torch.cuda.synchronize()
        for value in values.values():
            validate_decisions(value)
        if any(not torch.equal(v, features.state_dict()[k]) for k, v in before.items()):
            raise ValueError('Query prediction changed frozen feature parameters or buffers')
        np.savez_compressed(output / 'predictions.npz', ids=ids, **values)
        d.write(output / 'complete.json', dict(status='PREDICTIONS_COMPLETE', config=c, count=COUNT,
                views=list(d.VIEWS), readouts=list(d.READOUTS), truth_read=False, query_fit=False,
                all_source_heads_frozen_before_target_IQ=True, baseline_original_reused=c['arm'] == 'LEGACY50',
                inference_seconds=dict(clean_all_readouts=time.perf_counter() - infer_started),
                wall_seconds=time.perf_counter() - started))
        print('PREDICTIONS_COMPLETE ' + json.dumps(dict(row_id=row, readouts=list(d.READOUTS), count=COUNT,
                truth_read=False, baseline_original_reused=c['arm'] == 'LEGACY50')), flush=True)


def preflight(row):
    import numpy as np
    c = d.config(next(r for r in d.rows() if r['row_id'] == row))
    source = source_proof(c)
    output = d.BASE / row / 'prediction'
    done = d.read(output / 'complete.json')
    required = dict(status='PREDICTIONS_COMPLETE', config=c, count=COUNT, views=list(d.VIEWS),
                    readouts=list(d.READOUTS), truth_read=False, query_fit=False,
                    all_source_heads_frozen_before_target_IQ=True, baseline_original_reused=c['arm'] == 'LEGACY50')
    if any(done.get(k) != v for k, v in required.items()):
        raise ValueError('Truth closed: incomplete fixed readout predictions')
    proof = d.read(output / 'provenance.json')
    expected = dict(status='VERIFIED', config=c, checkpoint=checkpoint_path(c), source_root=str(source),
                    source_roles='EXACT_MATCH', ancestors=[], own_final50=True,
                    old_W_discarded_for_new_readouts=True, heads_frozen_before_target_IQ=True,
                    readouts=list(d.READOUTS), truth_read=False, query_fit=False,
                    phase1_source_readouts=True, Phase2_claim=False, baseline_original_reused=c['arm'] == 'LEGACY50',
                    baseline_prediction_root=str(baseline_prediction_root(c)) if c['arm'] == 'LEGACY50' else None)
    if proof != expected:
        raise ValueError('Truth closed: readout/source provenance differs')
    validate_head_freeze(c, d.read(output / 'head_frozen.json'))
    if not (output / 'source_heads.pth').is_file():
        raise ValueError('Truth closed: frozen source classifier artifact missing')
    from experiments.cvs_frequency_context50.predict import observation_metadata
    ids, unused = observation_metadata()
    with np.load(output / 'predictions.npz', allow_pickle=False) as saved:
        if set(saved.files) != {'ids', *d.READOUTS} or not np.array_equal(saved['ids'], ids):
            raise ValueError('Truth closed: fixed predictions/IDs differ')
        for name in d.READOUTS:
            validate_decisions(saved[name])
        if c['arm'] == 'LEGACY50' and not np.array_equal(saved['ORIGINAL_HEAD'], baseline_original(c, ids)):
            raise ValueError('Truth closed: baseline original decisions changed instead of being reused')
    return c, ids
