from __future__ import annotations
import json
from pathlib import Path
from .legacy.options import build_arg_parser

SCENES = ('leo_clear_weak', 'leo_low_elev_weak', 'leo_rain_weak')

def parser():
    p = build_arg_parser()
    argv = json.loads((Path(__file__).resolve().parents[2] / 'configs/core90_game_historical_argv.json').read_text(encoding='utf-8'))
    defaults = vars(p.parse_args(argv))
    defaults.update(labeled_ratio=.07, unlabeled_ratio=.63, source_val_ratio=.30,
                    wisig_train_rxs='1,3,4,6,8', wisig_test_rxs='0,2,5,7,9,10,11',
                    wisig_train_days='1,2,3', wisig_test_days='0,1,2,3',
                    best_metric='clean_val_tx', enable_joint_safe_guard=False,
                    paic_guard_enabled=False, baseline_ckpt='', from_scratch=True,
                    amp=False, use_concat_sat_channel_aug=True, concat_sat_ce_only=True)
    p.set_defaults(**defaults)
    p.add_argument('--game_solver', choices=('simultaneous','alternating','extragradient','heun','optimistic','head_lookahead'), default='simultaneous')
    p.add_argument('--game_evidence_version',type=int,choices=(1,2),default=2)
    p.add_argument('--game_deterministic',action='store_true',help='Use the explicitly accepted deterministic backend path')
    p.add_argument('--game_b8_impl',choices=('reference','head_grad_only','graph_reuse'),default='reference')
    p.add_argument('--game_telemetry_interval',type=int,default=250)
    p.add_argument('--game_capability_interval',type=int,default=250)
    p.add_argument('--game_probe_lr',type=float,default=.002)
    p.add_argument('--game_capability_lr',type=float,default=.002)
    p.add_argument('--game_data_order_seed',type=int,default=-1)
    p.add_argument('--game_data_contract',type=json.loads,default={})
    p.add_argument('--game_exposure_replay',default='')
    p.add_argument('--game_optimizer', choices=('adamw','sgd'), default='adamw')
    p.add_argument('--game_control', choices=('off','catchup','correction','both','random','fixed','replay'), default='off')
    p.add_argument('--game_curriculum', choices=('fixed','capability'), default='fixed')
    p.add_argument('--game_head_scale', choices=('legacy_weighted','separate_head_scale'), default='legacy_weighted')
    p.add_argument('--game_head_lr_ratio', type=float, default=1.)
    p.add_argument('--game_fixed_head_steps', type=int, default=0)
    p.add_argument('--game_audit_interval', type=int, default=250)
    p.add_argument('--game_audit_samples_per_capture', type=int, default=4)
    p.add_argument('--game_probe_steps', type=int, default=40)
    p.add_argument('--game_independent_probe_every', type=int, default=4)
    p.add_argument('--game_max_extra_head', type=int, default=3)
    p.add_argument('--game_correction_fraction', type=float, default=.20)
    p.add_argument('--game_actions_replay', default='')
    p.add_argument('--game_jacobian_interval', type=int, default=0)
    p.add_argument('--game_response_tracking', action='store_true')
    p.add_argument('--game_max_grad_norm', type=float, default=5.)
    p.add_argument('--game_optimistic_pseudo_change',type=float,default=.25)
    p.add_argument('--game_pseudo_change_window',type=int,default=20)
    p.add_argument('--game_max_steps_per_epoch', type=int, default=0)
    p.add_argument('--game_time_budget_s', type=float, default=0.)
    p.add_argument('--game_resume', default='')
    p.add_argument('--game_synthetic', action='store_true')
    p.add_argument('--game_split_seed', type=int, default=392002)
    p.add_argument('--game_source_calibration_steps', type=int, default=500)
    p.add_argument('--game_no_audit', action='store_true')
    p.add_argument('--game_export_source_predictions', action='store_true')
    p.add_argument('--game_skip_final_eval', action='store_true')
    p.add_argument('--game_config_json', default='')
    return p

def parse_args(argv=None):
    p = parser()
    pre, _ = p.parse_known_args(argv)
    if pre.game_config_json:
        overrides = json.loads(Path(pre.game_config_json).read_text(encoding='utf-8'))
        known = {a.dest for a in p._actions}
        if set(overrides) - known:
            raise ValueError('Unknown configuration keys: ' + str(set(overrides) - known))
        p.set_defaults(**overrides)
    args = p.parse_args(argv)
    validate(args)
    return args

def validate(a):
    if a.game_evidence_version==2 and a.game_control in ('fixed','random'):
        raise ValueError('V2 requires independent source donor frozen replay; online fixed/random is legacy v1 only')
    if a.game_evidence_version==2 and a.game_no_audit and (
            a.game_control!='off' or a.game_curriculum=='capability' or
            a.game_response_tracking or a.game_jacobian_interval):
        raise ValueError('Requested V2 mechanism requires source audits; remove --game_no_audit')
    if a.baseline_ckpt or not a.from_scratch:
        raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED: initial training must be scratch-only')
    if a.best_metric != 'clean_val_tx' or a.enable_joint_safe_guard or a.paic_guard_enabled:
        raise ValueError('Source-only fixed-final selection required; legacy test guards are prohibited')
    if [a.labeled_ratio, a.unlabeled_ratio, a.source_val_ratio] != [.07,.63,.30]:
        raise ValueError('Phase1 source roles must be .07/.63/.30')
    if not a.use_concat_sat_channel_aug or not a.concat_sat_ce_only or a.lambda_sat_cons != 0:
        raise ValueError('CORE90 matched rows require concat satellite CE-only')
    source = set(a.wisig_train_rxs.split(','))
    if source & set(a.wisig_test_rxs.split(',')):
        raise ValueError('source/target RX overlap')
    if a.game_control == 'replay' and not a.game_actions_replay:
        raise ValueError('replay requires a previously source-generated schedule')
    if a.game_max_steps_per_epoch and not a.game_synthetic:
        raise ValueError('Step truncation is restricted to synthetic functional acceptance')
    if a.game_fixed_head_steps < 0 or a.game_max_extra_head < 0 or a.game_probe_steps < 1 or a.game_head_lr_ratio <= 0:
        raise ValueError('Invalid update budget')
    if a.epochs <= 0:
        raise ValueError('epochs must be positive')
    if not 0<a.game_optimistic_pseudo_change<=1 or a.game_pseudo_change_window<1:
        raise ValueError('Invalid pseudo distribution change window/threshold')
    if a.game_evidence_version not in (1,2) or a.game_b8_impl not in ('reference','head_grad_only','graph_reuse'):
        raise ValueError('Unknown game evidence or B8 implementation version')
    if a.game_solver=='head_lookahead' and a.game_b8_impl=='graph_reuse' and a.amp:
        raise ValueError('graph_reuse is validated for FP32 only; AMP is not supported')
    if a.game_capability_interval<1 or a.game_audit_interval<1 or min(a.game_probe_lr,a.game_capability_lr)<=0:
        raise ValueError('Invalid independent audit/capability configuration')
    if a.game_telemetry_interval<0 or a.game_data_order_seed < -1 or not isinstance(a.game_data_contract,dict):
        raise ValueError('Invalid telemetry/data contract configuration')
    if a.game_exposure_replay and (a.game_evidence_version!=2 or a.game_curriculum!='fixed'):
        raise ValueError('Frozen exposure replay requires v2 and fixed curriculum')
