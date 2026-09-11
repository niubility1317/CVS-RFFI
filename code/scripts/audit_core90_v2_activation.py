"""Read and resolve persisted configurations; do not launch training."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
from cvsrffi.game_tracking.config import parse_args
from cvsrffi.game_tracking.legacy.options import _loss_weights
from cvsrffi.schedule import build_stage_state
from scripts.build_core90_game_matrix import build_matrix


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', required=True)
    out = Path(cli.parse_args().output)
    fields = ('game_evidence_version','game_solver','game_b8_impl','game_control',
        'game_curriculum','game_no_audit','game_telemetry_interval','game_probe_lr',
        'game_probe_steps','game_capability_lr','game_audit_interval',
        'game_capability_interval','game_deterministic','game_response_tracking',
        'game_jacobian_interval','lambda_adv','lambda_sat_cls','lambda_sat_cons',
        'sat_cons_start_epoch','label_epochs','epochs','lr','game_head_lr_ratio',
        'use_unlabeled','use_ema_teacher','amp','from_scratch','baseline_ckpt','game_resume')
    rows = []
    for folder in ('core90_game_v2_planned_20260911',
                   'core90_game_v2_strong_source_planned_20260911'):
        manifest = json.loads((ROOT / 'configs' / folder / 'matrix.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'CONFIGURED_NOT_LAUNCHED'
        for run in manifest['runs']:
            p = ROOT / 'configs' / folder / (run['run_id'] + '.json')
            raw = json.loads(p.read_text(encoding='utf-8'))
            assert raw == run['config'], run['run_id']
            args = parse_args(['--output_dir', raw['output_dir'], '--game_config_json', str(p)])
            resolved = {k:getattr(args,k) for k in fields}
            assert resolved['game_evidence_version'] == 2
            assert args.game_control == 'off' and args.game_curriculum == 'fixed' and args.game_no_audit
            assert args.game_deterministic and not args.amp
            assert args.from_scratch and not args.baseline_ckpt and not args.game_resume
            assert args.epochs == 200 and args.label_epochs == 130 and args.sat_cons_start_epoch == 80
            rows.append(dict(row=run['row'],seed=run['seed'],config=p.relative_to(ROOT).as_posix(),
                resolved=resolved,stages={str(e):_loss_weights(args,build_stage_state(e,args))
                    for e in (1,40,41,79,80,90,91,130,131,200)}))
    assert len(rows) == 21
    expected = {'V2_A':('simultaneous',0.),'V2_B':('simultaneous',.35),
        'V2_C':('head_lookahead',0.),'V2_D':('head_lookahead',.35),
        'V2_E':('extragradient',0.),'V2_F':('extragradient',.35)}
    for r in rows:
        if r['row'] in expected:
            a = r['resolved']
            assert (a['game_solver'],a['lambda_adv']) == expected[r['row']]
            assert a['game_b8_impl'] == ('head_grad_only' if a['game_solver']=='head_lookahead' else 'reference')
    old = build_matrix(rows=('B8','S4','C2'), seeds=(392005,))
    defaults = parse_args(['--output_dir','not_launched'])
    selected = parse_args(['--output_dir','not_launched','--game_config_json',
                          str(ROOT/'configs/core90_game_v2_selected_probe_20260911.json')])
    report = dict(status='VERIFIED_CONFIG_ONLY_NOT_RUNNING',rows=rows,
        bare_cli_defaults={k:getattr(defaults,k) for k in fields},
        selected_probe={k:getattr(selected,k) for k in ('game_evidence_version','game_probe_lr','game_probe_steps')},
        legacy_named_rows=[dict(row=r['row'],evidence_version=r['config']['game_evidence_version']) for r in old],
        interpretation='F1 intentionally disables audit/control/capability; legacy B8/S4/C2 row names remain V1. No new formal run checked or launched.')
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],rows=len(rows),legacy_rows=report['legacy_named_rows'])))


if __name__ == '__main__':
    main()
