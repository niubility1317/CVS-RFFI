"""Resolve the CORE90 ablation matrix; this utility never launches training."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

VARIANTS = ('U0', 'U1', 'U2', 'U3', 'U4_additive', 'U4_bilinear', 'U5', 'Ux', 'head_only', 'permanent_detach')
V2_VARIANTS = VARIANTS + ('U1_mask_off', 'U3_delta', 'U3_delta_pairs', 'Ux_normalized',
                         'U4_decomposed', 'U5_reliable')
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'configs' / 'phase1_core90_cross_response_v1.json'
STORE_FLAGS = {'use_sat_consistency', 'use_concat_sat_channel_aug', 'concat_sat_ce_only', 'use_crra'}
ROLE_FIELDS = ('wisig_train_rxs', 'wisig_test_rxs', 'wisig_train_days', 'wisig_test_days')


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding='utf-8') as handle:
        config = json.load(handle)
    if config.get('schema_version') not in (1, 2):
        raise ValueError('unsupported cross-response schema_version')
    if set(config.get('variants', {})) != set(V2_VARIANTS if config['schema_version'] == 2 else VARIANTS):
        raise ValueError('the complete registered ablation matrix is required')
    return config


def resolve_variant(config: dict[str, Any], variant: str) -> dict[str, Any]:
    if variant not in config['variants']:
        raise ValueError(f'unknown cross-response variant: {variant}')
    result = copy.deepcopy(config['cross_response'])
    overrides = config['variants'][variant]
    unknown = set(overrides) - set(result)
    if unknown:
        raise ValueError(f'unknown variant keys: {sorted(unknown)}')
    result.update(overrides)
    for name in ('P', 'Q', 'K', 'target_dim', 'rank', 'readout_dim', 'update_interval'):
        if not isinstance(result[name], int) or isinstance(result[name], bool) or result[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
    if result['enabled'] and min(result['P'], result['Q']) < 3:
        raise ValueError('registered matrix requires independent 2x2 query plus donors')
    for name in ('lambda_resp', 'lambda_dec', 'lambda_cross', 'interaction_weight', 'gradient_cap'):
        value = result[name]
        if not isinstance(value, (float, int)) or not 0 <= value < float('inf'):
            raise ValueError(f'{name} must be finite and nonnegative')
    if result['predictor_mode'] not in {'additive', 'bilinear'}:
        raise ValueError('invalid predictor_mode')
    if result['scheduler_mode'] not in {'uniform', 'feedback'}:
        raise ValueError('invalid scheduler_mode')
    if not 0 < result['exploration'] <= 1:
        raise ValueError('exploration must be in (0,1]')
    if result['view_policy'] != 'clean_only':
        raise ValueError('first version permits only labeled clean auxiliary records')
    if not result['enabled'] and any(result[k] for k in ('response_enabled', 'decision_enabled', 'identity_interaction_enabled', 'head_only', 'permanent_detach')):
        raise ValueError('disabled auxiliary package cannot enable mechanisms')
    if result['identity_interaction_enabled'] and (result['response_enabled'] or result['decision_enabled']):
        raise ValueError('Ux direct interaction control must remain separate')
    if result['head_only'] and result['decision_enabled']:
        raise ValueError('head_only cannot update the real decision path')
    return result


def _ids(value: str, name: str) -> set[int]:
    try:
        values = [int(item.strip()) for item in value.split(',') if item.strip()]
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f'{name} requires comma-separated physical integer IDs') from exc
    if not values or min(values) < 0 or len(values) != len(set(values)):
        raise ValueError(f'{name} must explicitly name unique nonnegative IDs')
    return set(values)


def validate_input_roles(roles: dict[str, str]) -> None:
    parsed = {name: _ids(roles.get(name, ''), name) for name in ROLE_FIELDS}
    if parsed['wisig_train_rxs'] & parsed['wisig_test_rxs']:
        raise ValueError('source/target physical receiver sets must be disjoint')


def validate_baseline(args: dict[str, Any]) -> None:
    required = dict(from_scratch=True, baseline_ckpt='', epochs=200, label_epochs=130,
                    pseudo_epochs=70, labeled_ratio=.07, unlabeled_ratio=.63,
                    source_val_ratio=.30, checkpoint_selection='final_only',
                    best_metric='clean_val_tx', phase1_source_val_selection_only=True,
                    enable_joint_safe_guard=False, phase2_export_prototypes=False)
    for key, value in required.items():
        if args.get(key) != value:
            raise ValueError(f'CORE90 input/selection contract requires {key}={value!r}')
    for key in ('resume_checkpoint', 'phase2_export_checkpoint'):
        if args.get(key):
            raise ValueError('matrix rows must initialize from scratch without external checkpoints')


def arguments_to_argv(values: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in values.items():
        if key in STORE_FLAGS:
            argv.append('--' + (key if value else 'no_' + key))
        else:
            argv.extend(['--' + key, str(value).lower() if isinstance(value, bool) else str(value)])
    return argv


def build_matrix(*, config_path: str | Path = DEFAULT_CONFIG, wisig_pkl: str,
                 output_root: str | Path, roles: dict[str, str], seeds: list[int],
                 variants: tuple[str, ...] | list[str] = VARIANTS) -> list[dict[str, Any]]:
    config = load_config(config_path)
    validate_input_roles(roles)
    validate_baseline(config['baseline_args'])
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('provide unique model seeds')
    if len(variants) != len(set(variants)):
        raise ValueError('duplicate variants would share output paths')
    rows = []
    for seed in seeds:
        for variant in variants:
            baseline = copy.deepcopy(config['baseline_args'])
            baseline.update(roles)
            baseline.update(wisig_pkl=wisig_pkl, seed=seed,
                            output_dir=str(Path(output_root) / f'{variant}_s{seed}'),
                            run_id=f"core90_cross_response_v{config['schema_version']}", candidate_id=f'{variant}_s{seed}')
            resolved = resolve_variant(config, variant)
            argv = arguments_to_argv(baseline)
            argv += ['--cross_response_config', str(Path(config_path).resolve()), '--cross_response_variant', variant]
            rows.append(dict(variant=variant, model_seed=seed, baseline_args=baseline,
                             cross_response=resolved, argv=argv, launch=False))
    return rows


def build_confirmation_matrix(*, confirmation_boundary, **kwargs):
    """Register paired multi-seed confirmation; this function has no launcher.

    The boundary is a user-frozen future evaluation plan, not inferred from the
    already-scored historical target. Empty or reused confirmation is rejected.
    """
    seeds = kwargs.get("seeds", [])
    if len(seeds) < 2:
        raise ValueError("confirmation requires explicitly named multiple model seeds")
    if (not isinstance(confirmation_boundary, dict)
            or confirmation_boundary.get("previous_target_results_used") is not False
            or not confirmation_boundary.get("independent_confirmation_scope")
            or not confirmation_boundary.get("frozen_before_launch")):
        raise ValueError("an explicit independent confirmation boundary must be frozen before launch")
    rows = build_matrix(**kwargs)
    if any(r["cross_response"].get("implementation_version") != 2 for r in rows):
        raise ValueError("V2 confirmation requires versioned V2 candidate configuration")
    for row in rows:
        row.update(confirmation_boundary=copy.deepcopy(confirmation_boundary),
            data_seed=row["cross_response"]["data_seed"],
            comparison_unit="paired model seed; query records are not training replicates",
            required_reporting=["clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak",
                                "RX/day lower tail", "per TX", "training and audit cost"],
            launch=False)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--wisig-pkl', required=True)
    parser.add_argument('--output-root', required=True)
    for name in ROLE_FIELDS:
        parser.add_argument('--' + name.replace('_', '-'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[392002])
    parser.add_argument('--variants', choices=V2_VARIANTS, nargs='+', default=list(VARIANTS))
    parser.add_argument('--write', type=Path)
    args = parser.parse_args()
    rows = build_matrix(config_path=args.config, wisig_pkl=args.wisig_pkl, output_root=args.output_root,
                        roles={name: getattr(args, name) for name in ROLE_FIELDS}, seeds=args.seeds,
                        variants=args.variants)
    payload = json.dumps(rows, ensure_ascii=False, indent=2) + '\n'
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        with args.write.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
