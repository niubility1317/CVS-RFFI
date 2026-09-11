"""Emit source-development configs and commands; never launch experiments."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.game_tracking.config import parser as training_parser, validate

DEFAULT_ROWS = ("B0", "B1", "B2", "B4", "B5", "B6", "S1", "S3", "S4", "C1", "C2")
DEFAULT_SEEDS = (392002, 392003)
V2_ROWS = tuple('V2_' + row for row in 'ABCDEF')
V2_SEEDS = (392005, 392006, 392007)
STRONG_SOURCE_ROWS = ('V2_STRONG_SOURCE_LR_LOW','V2_STRONG_SOURCE_LR_BASE','V2_STRONG_SOURCE_LR_HIGH')
STRONG_SELECTION_RULE = dict(schema='core90_strong_ordinary_selection_v2',
    metric='unweighted_mean_four_source_scene_macro_f1',checkpoint='fixed_final_E200',
    tie_break='lower_learning_rate',candidate_lrs=[1e-4,2e-4,4e-4],
    selection_scope='one_paired_source_seed_once_then_same_configuration_all_confirmation_seeds',target_allowed=False)
REAUDIT_PRIORITY = ('J1','H1','B7','B8','C2','S4','S3','S1','C1')


def row_menu(replay_path=""):
    """B3 candidates require source-only selection; none is claimed tuned."""
    rows = {
        "B0": ("matched CORE90 scratch baseline without audits", {"game_no_audit": True}),
        "B1": ("source audits with ordinary updates", {}),
        "B2": ("identity-domain adversarial field disabled; domain retention remains", {"lambda_adv": 0.}),
        "B3": ("source learning-rate candidate: half baseline", {"lr": 1e-4}),
        "B3_lr_high": ("source learning-rate candidate: twice baseline", {"lr": 4e-4}),
        "B3_head_slow": ("source head-timescale candidate", {"game_head_lr_ratio": .5}),
        "B3_head_fast": ("source head-timescale candidate", {"game_head_lr_ratio": 2.}),
        "B3_head_scale": ("separate head CE scale; encoder keeps adversarial coefficient", {"game_head_scale":"separate_head_scale"}),
        "B3_adv_low": ("source adversarial-scale candidate", {"lambda_adv": .175}),
        "B3_adv_high": ("source adversarial-scale candidate", {"lambda_adv": .7}),
        "B4": ("alternating with two total head steps", {"game_solver": "alternating", "game_fixed_head_steps": 2}),
        "B4_fixedk": ("fixed head catch-up plus ordinary coupled update", {"game_fixed_head_steps": 2}),
        "B5": ("complete extragradient", {"game_solver": "extragradient"}),
        "B6": ("Heun average-gradient update", {"game_solver": "heun"}),
        "B7": ("optimistic field update with event resets", {"game_solver": "optimistic"}),
        "B8": ("head-only predictor approximation", {"game_solver": "head_lookahead"}),
        "S1": ("source-triggered bounded head catch-up", {"game_control": "catchup"}),
        "S2": ("fixed correction schedule cost control", {"game_control": "fixed"}),
        "S2_random": ("random correction schedule cost control", {"game_control": "random"}),
        "S3": ("source-triggered correction", {"game_control": "correction"}),
        "S4": ("source-triggered catch-up and correction", {"game_control": "both"}),
        "C1": ("source capability curriculum with ordinary updates", {"game_curriculum": "capability"}),
        "C2": ("source capability curriculum and both controls", {"game_control": "both", "game_curriculum": "capability"}),
        "C3": ("capability curriculum with random correction timing", {"game_control": "random", "game_curriculum": "capability"}),
        "H1": ("optional implicit head-response experiment", {"game_response_tracking": True}),
        "J1": ("full active-objective local Jacobian diagnostic", {"game_jacobian_interval":500}),
    }
    for row, solver, adv in zip(V2_ROWS,
            ('simultaneous', 'simultaneous', 'head_lookahead', 'head_lookahead', 'extragradient', 'extragradient'),
            (0., .35, 0., .35, 0., .35)):
        rows[row] = ('V2 paired solver x identity adversarial factor; fixed old course; no audits',
                     dict(game_solver=solver, lambda_adv=adv, game_no_audit=True,
                          game_curriculum='fixed', game_evidence_version=2,game_deterministic=True,
                          game_b8_impl='head_grad_only' if solver == 'head_lookahead' else 'reference'))
    for row, lr in zip(STRONG_SOURCE_ROWS,(1e-4,2e-4,4e-4)):
        rows[row]=('Independent source-only ordinary LR candidate, not a confirmed strong baseline',
            dict(game_solver='simultaneous',lambda_adv=.35,lr=lr,game_no_audit=True,
                 game_curriculum='fixed',game_evidence_version=2,game_deterministic=True,game_b8_impl='reference'))
    if replay_path:
        rows["C4"] = ("cross-seed replay of a frozen source action schedule",
                      {"game_control": "replay", "game_actions_replay": str(replay_path)})
    return rows


def build_matrix(*, rows=DEFAULT_ROWS, seeds=DEFAULT_SEEDS, runs_root="runs/core90_game",
                 replay_path="", dataset_path=None, device="cuda:0"):
    if not seeds or len(set(seeds)) != len(seeds) or any(int(seed) < 0 for seed in seeds):
        raise ValueError("Specify distinct training seeds")
    if not rows or len(set(rows)) != len(rows):
        raise ValueError("Specify distinct experiment rows")
    menu = row_menu(replay_path)
    unknown = set(rows) - set(menu)
    if unknown:
        raise ValueError(f"Unknown or unavailable rows {sorted(unknown)}; C4 requires explicit source replay path")
    result = []
    p = training_parser()
    known = {action.dest for action in p._actions}
    for row in rows:
        description, override = menu[row]
        for seed in seeds:
            run_id = f"{row}_seed{int(seed)}"
            config = {
                "epochs": 200, "label_epochs": 130, "pseudo_epochs": 70,
                "seed": int(seed), "game_split_seed": 392002,
                "from_scratch": True, "baseline_ckpt": "", "game_resume": "",
                "labeled_ratio": .07, "unlabeled_ratio": .63, "source_val_ratio": .30,
                "game_solver": "simultaneous", "game_optimizer": "adamw",
                "game_control": "off", "game_curriculum": "fixed", "game_no_audit": False,
                "game_evidence_version": 1,
                "game_fixed_head_steps": 0, "game_response_tracking": False,
                "game_actions_replay": "", "game_max_steps_per_epoch": 0,
                "game_synthetic": False, "game_skip_final_eval": False,
                "game_export_source_predictions": True, "game_correction_fraction": .20,
                "output_dir": (Path(runs_root) / run_id).as_posix(), "run_name": run_id,
                "metrics_csv": "", "metrics_jsonl": "", "device": device,
            }
            if dataset_path is not None:
                config["wisig_pkl"] = str(dataset_path)
            config.update(override)
            if set(config) - known:
                raise ValueError("Matrix used unknown training configuration keys")
            args = p.parse_args(["--output_dir", config["output_dir"]])
            for key, value in config.items():
                setattr(args, key, value)
            validate(args)
            result.append({"row": row, "run_id": run_id, "seed": seed,
                           "description": description, "config": config})
    return result


def select_strong_ordinary(candidate_runs):
    """Select once from complete, paired source final scores, never target files."""
    if len(candidate_runs)!=3 or {r['row'] for r in candidate_runs}!=set(STRONG_SOURCE_ROWS):
        raise ValueError('exactly the three preregistered strong source candidates required')
    if len({r['seed'] for r in candidate_runs})!=1:
        raise ValueError('strong source selection candidates must share one seed')
    scored=[];missing=[];configs=[]
    for run in candidate_runs:
        root=Path(run['config']['output_dir'])
        files=[root/'resolved_config.json',root/'completion.json',root/'source_final_eval/source_scores.json']
        if not all(p.exists() for p in files): missing.append(run['row']);continue
        config,completion,scores=[json.loads(p.read_text(encoding='utf-8-sig')) for p in files]
        if (completion.get('status')!='SOURCE_ARTIFACTS_COMPLETE' or completion.get('epoch')!=200
                or completion.get('target_evaluated') is not False or scores.get('complete') is not True
                or scores.get('source_only') is not True):
            missing.append(run['row']);continue
        required=dict(game_solver='simultaneous',game_curriculum='fixed',game_no_audit=True,
                      epochs=200,from_scratch=True,baseline_ckpt='',game_resume='',lambda_adv=.35,game_evidence_version=2,
                      game_deterministic=True)
        if any(config.get(k)!=v for k,v in required.items()) or config.get('seed')!=run['seed'] or config.get('lr')!=run['config']['lr']:
            raise ValueError('strong source candidate deviates from preregistered configuration')
        if config.get('game_max_steps_per_epoch',0) or config.get('game_synthetic',False):
            raise ValueError('strong source selection requires complete real-source candidates')
        values=[scores.get('scenes',{}).get(scene,{}).get('macro_f1') for scene in
                ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')]
        if not all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1 for v in values):
            missing.append(run['row']);continue
        configs.append(config)
        scored.append(dict(row=run['row'],seed=run['seed'],lr=config['lr'],source_mean_macro_f1=sum(values)/4,
                           config=config,source_score_path=str(files[-1].resolve())))
    if missing:
        return dict(schema=STRONG_SELECTION_RULE['schema'],status='DEFERRED_INCOMPLETE_SOURCE_CANDIDATES',
                    selection_rule=STRONG_SELECTION_RULE,missing=missing,selection_performed=False)
    incidental={'lr','run_name','output_dir','game_config_json','game_row','metrics_csv','metrics_jsonl'}
    reference={k:v for k,v in configs[0].items() if k not in incidental}
    if any({k:v for k,v in c.items() if k not in incidental}!=reference for c in configs[1:]):
        raise ValueError('strong candidates differ beyond learning rate')
    selected=sorted(scored,key=lambda r:(-r['source_mean_macro_f1'],r['lr']))[0]
    return dict(schema=STRONG_SELECTION_RULE['schema'],status='SOURCE_SELECTED_NOT_CONFIRMED',selection_performed=True,
                selection_rule=STRONG_SELECTION_RULE,selected_lr=selected['lr'],selected_config=selected['config'],
                selected_row=selected['row'],source_seed=selected['seed'],candidates=scored,target_used=False)


def build_strong_confirmation(selection, *, seeds=V2_SEEDS, runs_root='runs/core90_game_strong_confirmation'):
    if selection.get('status')!='SOURCE_SELECTED_NOT_CONFIRMED' or selection.get('target_used') is not False or selection.get('selection_rule')!=STRONG_SELECTION_RULE:
        raise ValueError('complete preregistered source selection required before confirmation config')
    result=build_matrix(rows=('V2_B',),seeds=seeds,runs_root=runs_root)
    for run in result:
        run['row']='V2_STRONG_CONFIRMATION';run['run_id']=f"V2_STRONG_CONFIRMATION_seed{run['seed']}"
        # Freeze the selected source configuration across every confirmation seed.
        config=dict(selection['selected_config'])
        for key in ('game_config_json','game_row'):config.pop(key,None)
        config.update(seed=run['seed'],lr=selection['selected_lr'],run_name=run['run_id'],
                      output_dir=(Path(runs_root)/run['run_id']).as_posix(),metrics_csv='',metrics_jsonl='')
        run['config']=config
        run['description']='One source-selected ordinary configuration, frozen across confirmation seeds'
        run['source_selection_row']=selection['selected_row']
    return result


def write_matrix(output_dir, *, strong_selection=None, **kwargs):
    output = Path(output_dir)
    rows = (build_strong_confirmation(strong_selection,seeds=kwargs.get('seeds',V2_SEEDS),
                                     runs_root=kwargs.get('runs_root','runs/core90_game_strong_confirmation'))
            if strong_selection is not None else build_matrix(**kwargs))
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / (row["run_id"] + ".json") for row in rows]
    if any(path.exists() for path in destinations) or (output / "matrix.json").exists():
        raise FileExistsError("Matrix artifacts already exist; choose a new output directory")
    for row, destination in zip(rows, destinations):
        with destination.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(row["config"], stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        # Keep an argv array: schedulers must not interpolate it through a shell.
        row["config_path"] = str(destination.resolve())
        row["argv"] = ["code/SSDG/train_core90_game.py", "--output_dir", row["config"]["output_dir"],
                       "--game_config_json", str(destination.resolve())]
    manifest = {"schema": "core90_game_matrix_v2" if all(r['row'].startswith('V2_') for r in rows) else "core90_game_matrix_v1", "status": "CONFIGURED_NOT_LAUNCHED",
                "source_development_only": True, "target_evaluation": False,
                "checkpoint_selection": "fixed_final_E200", "split_seed": 392002,
                "budget_matching": "Compare realized field/head/audit counts and wall time; fractions alone are not matched cost",
                "initial_default_rows": list(DEFAULT_ROWS), "runs": rows}
    if any(r['row'] in STRONG_SOURCE_ROWS for r in rows):
        manifest['strong_source_selection_rule']=STRONG_SELECTION_RULE
        manifest['strong_candidate_status']='SOURCE_CANDIDATES_NOT_SELECTED_OR_CONFIRMED'
    if strong_selection is not None: manifest['strong_source_selection']=strong_selection
    if any(r['row'] in ('V2_C','V2_D') for r in rows):
        manifest['b8_implementation_acceptance']='PENDING_B8_LONG_TRAJECTORY_ACCEPTANCE'
    with (output / "matrix.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--runs-root", default="runs/core90_game")
    p.add_argument("--rows", default=",".join(DEFAULT_ROWS), help="Comma-separated IDs or all")
    p.add_argument("--seeds", help='Explicit configured seeds; defaults to V2 proposal for --rows v2, legacy otherwise')
    p.add_argument("--replay-path", default="")
    p.add_argument("--dataset-path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument('--strong-source-matrix',help='Select from a complete source candidate matrix, then configure one frozen strong confirmation')
    a = p.parse_args(argv)
    if a.strong_source_matrix:
        source_manifest=json.loads(Path(a.strong_source_matrix).read_text(encoding='utf-8-sig'))
        if source_manifest.get('strong_source_selection_rule')!=STRONG_SELECTION_RULE:
            p.error('candidate matrix does not contain the preregistered source selection rule')
        selection=select_strong_ordinary(source_manifest['runs'])
        output=Path(a.output_dir);output.mkdir(parents=True,exist_ok=True)
        with (output/'strong_source_selection.json').open('x',encoding='utf-8') as handle:
            json.dump(selection,handle,indent=2,allow_nan=False)
        if not selection['selection_performed']:
            print(json.dumps({'status':selection['status'],'missing':selection['missing']}));return
        manifest=write_matrix(output,strong_selection=selection,
            seeds=tuple(int(v) for v in a.seeds.split(',')) if a.seeds else V2_SEEDS,runs_root=a.runs_root)
        print(json.dumps({'status':manifest['status'],'runs':len(manifest['runs'])}));return
    rows = (V2_ROWS if a.rows == 'v2' else STRONG_SOURCE_ROWS if a.rows=='v2_strong_source' else tuple(row_menu(a.replay_path)) if a.rows == "all" else
            REAUDIT_PRIORITY+tuple(r for r in row_menu(a.replay_path) if r not in REAUDIT_PRIORITY)
            if a.rows=='reaudit' else tuple(a.rows.split(",")))
    seeds = tuple(int(v) for v in a.seeds.split(',')) if a.seeds else V2_SEEDS if a.rows == 'v2' else (392005,) if a.rows=='v2_strong_source' else DEFAULT_SEEDS
    manifest = write_matrix(a.output_dir, rows=rows, seeds=seeds,
                            runs_root=a.runs_root, replay_path=a.replay_path,
                            dataset_path=a.dataset_path, device=a.device)
    print(json.dumps({"status": manifest["status"], "runs": len(manifest["runs"]),
                      "manifest": str((Path(a.output_dir) / "matrix.json").resolve())}))


if __name__ == "__main__":
    main()
