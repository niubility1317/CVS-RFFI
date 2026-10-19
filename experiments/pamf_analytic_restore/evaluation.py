import json
from types import SimpleNamespace
from . import design as d
from .runtime import load_checkpoint
from experiments.pamf_uncertainty_restore.evaluation import VIEWS, predict as original_predict, score as original_score
from experiments.pamf_native_formal.runner import assess_goal
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_restoration.data import write_json


def row_args(args):
    return SimpleNamespace(**dict(vars(args), output=args.output/args.row, kind=d.KIND))


def prepare_target(args):
    decision = read_json(args.output/'source_decision.json')
    if decision['status'] != 'SOURCE_DECISION_FROZEN' or not decision['qualified'] or args.row not in decision['target_rows']:
        raise ValueError('Primary analytic source qualification required before target data')
    root = args.output/args.row
    write_json(root/'source_decision.json', dict(decision, target_candidates=[d.KIND]))


def predict(args, role):
    if role == 'target':
        prepare_target(args)
    def loader(path, device):
        model, payload = load_checkpoint(path, device)
        if payload['candidate'] != args.row:
            raise ValueError('Checkpoint belongs to another analytic comparison row')
        return model, payload
    original_predict(row_args(args), role, checkpoint_loader=loader)


def score(args, role):
    original_score(row_args(args), role)
    if role == 'target':
        root = args.output/args.row
        summary = read_json(root/'target_confirmation'/d.KIND/'summary.json')
        write_json(root/'goal_assessment.json', assess_goal(summary))


def qualify_source(control, analytic, parent):
    import math
    for item in (control, analytic, parent):
        expected = dict(status='ANALYZED', kind=d.KIND, role='source', count=1800, F_run=REFERENCE_RUN,
            F_seed=d.MODEL_SEED, original_default_head=True, all_predictions_before_truth=True,
            training_TX_labels_read=0, training_F_access=False, head_fit_updates=0, target_access=False)
        if any(item.get(key) != value for key, value in expected.items()) or list(item['views']) != list(VIEWS):
            raise ValueError('Complete same-source original-F50 evidence required')
    def accuracy(item, view, dimension='overall'):
        rows = [x for x in item['records'] if x['view']==view and x['dimension']==dimension and x['readout']=='restored']
        if len({x['value'] for x in rows}) != len(rows) or any(not math.isfinite(x['accuracy']) or not 0 <= x['accuracy'] <= 100 for x in rows):
            raise ValueError('Unique finite source class groups required')
        return {x['value']: x['accuracy'] for x in rows}
    def mean(item):
        values = [item['views'][view]['restored']['waveform']['iq_nmse'] for view in VIEWS[1:]]
        if any(not math.isfinite(x) or x < 0 for x in values):
            raise ValueError('Finite nonnegative source waveform metrics required')
        return sum(values)/6
    mean_c, mean_a, mean_p = mean(control), mean(analytic), mean(parent)
    clean_nmse = analytic['views']['clean']['restored']['waveform']['iq_nmse']
    if not math.isfinite(clean_nmse) or clean_nmse < 0:
        raise ValueError('Finite clean NMSE required')
    gains = {}
    for view in ('practical_low_urban', 'practical_mid_urban'):
        value = accuracy(analytic, view)['all']
        gains[view] = dict(vs_control_pp=value-accuracy(control, view)['all'], vs_parent_pp=value-accuracy(parent, view)['all'])
    rows = [x for x in analytic['records'] if x['view']=='clean' and x['dimension']=='overall' and x['readout']=='restored']
    original_clean = rows[0]['baseline_accuracy']
    if not math.isfinite(original_clean) or not 0 <= original_clean <= 100:
        raise ValueError('Finite original source clean reference required')
    clean_drop = original_clean-accuracy(analytic, 'clean')['all']
    new_tx, old_tx = accuracy(analytic,'practical_low_urban','TX'),accuracy(parent,'practical_low_urban','TX')
    if set(new_tx) != {str(i) for i in range(6)} or set(new_tx) != set(old_tx):
        raise ValueError('All six source TX groups required')
    worst_tx = min(new_tx[key]-old_tx[key] for key in new_tx)
    other_delta = min(accuracy(analytic,view)['all']-accuracy(parent,view)['all'] for view in VIEWS[1:] if view!='practical_low_urban')
    wave_pass = mean_a <= 1.05*mean_c and mean_a <= 1.05*mean_p and clean_nmse <= .02
    identity_pass = (all(item['vs_control_pp'] >= .5 and item['vs_parent_pp'] >= .5 for item in gains.values())
        and clean_drop <= .1 and worst_tx >= -1. and other_delta >= -.5)
    qualified = bool(wave_pass and identity_pass)
    return dict(status='SOURCE_DECISION_FROZEN',qualified=qualified,waveform_A=bool(wave_pass),source_identity_B=bool(identity_pass),
        mean_control=mean_c,mean_analytic=mean_a,mean_parent=mean_p,analytic_clean_NMSE=clean_nmse,identity_gains=gains,
        clean_drop_vs_original_pp=clean_drop,worst_lowurban_TX_delta_pp=worst_tx,worst_other_scene_delta_pp=other_delta,
        target_rows=list(d.ROWS) if qualified else [],training_TX_F_access=False,identity_B_uses_source_V_labels=True,
        target_reads_before_decision=0,target_feedback=False,thresholds=d.configuration()['source_thresholds'])
