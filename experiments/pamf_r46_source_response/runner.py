import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback
import torch
from . import design as d
from .metrics import center_metrics, perturbations, power, response_fields, response_metrics, step_disagreement
from .runtime import check_frozen, load_model, validate_sources
from experiments.pamf_r46_source_fidelity.runner import load_panel
from experiments.pamf_r46_source_fidelity.metrics import summarize
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import code_commit


def grouped(values):
    if any(len(v) != 2*d.PER_ROLE for v in values.values()):
        raise ValueError('Complete fixed source panel required')
    return {role:dict(count=len(next(iter(values.values()))[sl]),
        metrics=summarize({key:value[sl] for key,value in values.items()}))
        for role,sl in (('all',slice(None)),('L_s',slice(0,d.PER_ROLE)),('U_s',slice(d.PER_ROLE,None)))}


def append(parts, measured):
    for key,value in measured.items():
        parts.setdefault(key, []).append(value.detach().cpu())


def finished(parts):
    return grouped({key:torch.cat(value) for key,value in parts.items()})


def restore(model, value, counts, kind):
    if value.ndim != 3 or value.shape[1:] != (2,256) or not torch.isfinite(value).all():
        raise ValueError('Finite native source IQ required')
    output = model(value)
    if output.shape != value.shape or not torch.isfinite(output).all():
        raise FloatingPointError('Full frozen restorer produced invalid output')
    counts[kind] += len(value)
    return output


@torch.no_grad()
def diagnose(args):
    if args.output.name != d.RUN or args.output.exists():
        raise ValueError('New immutable diagnostic output required')
    if code_commit() != args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Matching committed code and CPU-only environment required')
    started = time.perf_counter()
    validate_sources(args)
    models, payloads = {}, {}
    for arm in d.ARMS:
        models[arm], payloads[arm] = load_model(args, arm)
    panel, positions, roles = load_panel(args)
    if tuple(panel) != d.VIEWS or len(positions) != 2*d.PER_ROLE:
        raise ValueError('Unchanged R21 source-only panel required')
    args.output.mkdir()
    args._owns_output = True
    write_json(args.output/'launch.json', dict(status='SOURCE_RESPONSE_RUNNING', commit=args.expected_commit,
        pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv, CPU_threads=2, GPU_workers=0, time=time.time()))
    write_json(args.output/'comparison_plan.json', d.configuration())
    write_json(args.output/'panel.json', dict(opaque_role_positions=positions.tolist(), roles=roles.tolist(),
        unique_physical_positions=256, source_roles=['L_s','U_s'], TX_labels_read=0, F_reads=0, V_reads=0, target_reads=0))
    snapshots = {arm:{k:v.clone() for k,v in model.state_dict().items()} for arm,model in models.items()}
    counts = {arm:dict(smoke=0,center=0,perturbed=0) for arm in d.ARMS}
    rows = {}
    for arm,model in models.items():
        folder = args.output/arm
        folder.mkdir()
        restore(model, panel['clean'][:2], counts[arm], 'smoke')
        check_frozen(model, snapshots[arm])
        write_json(folder/'smoke.json', dict(status='PASS', real_L_s_IQ=2, F_reads=0,
            TX_labels_read=0, V_reads=0, target_reads=0, optimizer_updates=0))
        print(json.dumps(dict(event='REAL_SOURCE_CHECKPOINT_SMOKE_PASS', arm=arm)), flush=True)
        rows[arm] = dict(parameters=d.PARAMETERS[arm], source_final=[payloads[arm]['epoch'],payloads[arm]['updates']], views={})
    low, high = d.EPSILONS
    for view_index,view in enumerate(d.VIEWS):
        inputs = panel[view]
        parts = {arm:dict(center={},responses={(probe,step):{} for probe in d.PROBES for step in d.EPSILONS},
                          scaling={probe:{} for probe in d.PROBES}) for arm in d.ARMS}
        for offset in range(0,len(inputs),d.BATCH):
            value, clean = inputs[offset:offset+d.BATCH], panel['clean'][offset:offset+d.BATCH]
            centers = {arm:restore(model,value,counts[arm],'center') for arm,model in models.items()}
            for arm in d.ARMS:
                append(parts[arm]['center'], center_metrics(centers[arm],clean,centers[d.REFERENCE_ARM]))
            probes = perturbations(value, d.SEED + view_index*10000 + offset)
            odd = {arm:{} for arm in d.ARMS}
            for probe in d.PROBES:
                for step in d.EPSILONS:
                    delta = probes[probe][step]
                    fields = {}
                    for arm,model in models.items():
                        plus = restore(model,value+delta,counts[arm],'perturbed')
                        minus = restore(model,value-delta,counts[arm],'perturbed')
                        fields[arm] = response_fields(plus,minus,centers[arm])
                        odd[arm][probe,step] = fields[arm][0]
                    teacher_odd, teacher_even = fields[d.REFERENCE_ARM]
                    for arm in d.ARMS:
                        measured = response_metrics(*fields[arm],teacher_odd,teacher_even,delta,clean)
                        append(parts[arm]['responses'][probe,step], measured)
                teacher_scaling = step_disagreement(odd[d.REFERENCE_ARM][probe,low],
                    odd[d.REFERENCE_ARM][probe,high],low,high,value)
                for arm in d.ARMS:
                    append(parts[arm]['scaling'][probe],dict(
                        student_step_disagreement=step_disagreement(odd[arm][probe,low],odd[arm][probe,high],low,high,value),
                        teacher_step_disagreement=teacher_scaling))
        for arm in d.ARMS:
            measured = dict(center=finished(parts[arm]['center']),
                responses={probe:{f'{step:.3f}':finished(parts[arm]['responses'][probe,step])
                           for step in d.EPSILONS} for probe in d.PROBES},
                scaling={probe:finished(parts[arm]['scaling'][probe]) for probe in d.PROBES})
            rows[arm]['views'][view] = measured
            write_json(args.output/arm/(view+'_metrics.json'), measured)
            print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE',arm=arm,view=view,count=256,
                center_IQ_nmse=measured['center']['all']['metrics']['paired_clean_nmse']['mean'],
                seconds=time.perf_counter()-started)),flush=True)
    for arm,model in models.items():
        check_frozen(model,snapshots[arm])
        if counts[arm] != dict(smoke=2,center=768,perturbed=6144):
            raise ValueError('Actual source forward exposure differs from fixed plan')
        rows[arm].update(checkpoint_unchanged=True, optimizer_updates=0, actual_IQ_inputs=counts[arm],
            L_s_diagnostic_inputs=3456, U_s_diagnostic_inputs=3456, L_s_smoke_inputs=2)
        write_json(args.output/arm/'complete.json',dict(status='SOURCE_RESPONSE_COMPLETE',arm=arm,
            views=list(d.VIEWS),checkpoint_unchanged=True,actual_IQ_inputs=counts[arm]))
    result = dict(status='SOURCE_RESPONSE_COMPLETE',run=d.RUN,method_commit=args.expected_commit,
        configuration=d.configuration(),rows=rows,seconds=time.perf_counter()-started,
        F_weight_reads=0,F_output_reads=0,TX_label_reads=0,V_reads=0,target_reads=0,
        optimizer_updates=0,new_network_versions=0,all_checkpoint_states_unchanged=True)
    write_json(args.output/'results.json',result)
    write_json(args.output/'complete.json',dict(status=result['status'],arms=list(rows),views=list(d.VIEWS),
        method_commit=args.expected_commit,seconds=result['seconds'],all_checkpoint_states_unchanged=True))
    print(json.dumps(dict(event='SOURCE_RESPONSE_COMPLETE',seconds=result['seconds'])),flush=True)


def parser():
    value = argparse.ArgumentParser(description='CPU-only finite source IQ responses;no F,TX,V,target',allow_abbrev=False)
    for flag in ('compact-parent','factor-run','complex-run','parent-action','source-cache','output'):
        value.add_argument('--'+flag,type=Path,required=True)
    value.add_argument('--expected-commit',required=True)
    return value


def main():
    args = parser().parse_args()
    args._owns_output = False
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    try:
        diagnose(args)
    except Exception as exc:
        if args._owns_output and not (args.output/'failure.json').exists():
            write_json(args.output/'failure.json',dict(status='TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
