import argparse,json,os,sys,time,traceback
from pathlib import Path
import torch
from . import design as d
from .metrics import measure
from experiments.pamf_r46_source_response.runtime import check_frozen,load_model,validate_sources
from experiments.pamf_r46_source_fidelity.runner import load_panel
from experiments.pamf_r46_source_fidelity.metrics import summarize
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import code_commit


def grouped(values):
    if len(values)!=29 or any(v.shape!=(256,) or not torch.isfinite(v).all() for v in values.values()):raise ValueError('Full fixed per-example moment matrix required')
    return {role:dict(count=len(next(iter(values.values()))[sl]),metrics=summarize({k:v[sl] for k,v in values.items()}))
        for role,sl in (('all',slice(None)),('L_s',slice(0,128)),('U_s',slice(128,None)))}


def restore(model,value,counts,kind):
    output=model(value)
    if output.shape!=value.shape or not torch.isfinite(output).all():raise FloatingPointError('Invalid complete frozen source restorer output')
    counts[kind]+=len(value);return output


@torch.no_grad()
def diagnose(args):
    if args.output.name!=d.RUN or args.output.exists():raise ValueError('New immutable source diagnostic output required')
    if code_commit()!=args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Pinned committed CPU-only source diagnostic required')
    started=time.perf_counter();validate_sources(args);models={};payloads={}
    for arm in d.ARMS:models[arm],payloads[arm]=load_model(args,arm)
    panel,positions,roles=load_panel(args)
    if tuple(panel)!=d.VIEWS or len(positions)!=256 or roles.tolist()!=['L_s']*128+['U_s']*128:raise ValueError('Original fixed source-training panel required')
    args.output.mkdir();args._owns_output=True
    write_json(args.output/'launch.json',dict(status='SOURCE_MOMENTS_RUNNING',commit=args.expected_commit,pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,CPU_threads=2,GPU_workers=0,time=time.time()))
    write_json(args.output/'comparison_plan.json',d.configuration())
    write_json(args.output/'panel.json',dict(opaque_role_positions=positions.tolist(),roles=roles.tolist(),unique_physical_positions=256,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0))
    snapshots={a:{k:v.clone() for k,v in m.state_dict().items()} for a,m in models.items()};counts={a:dict(smoke=0,center=0) for a in d.ARMS};rows={}
    for arm,model in models.items():
        folder=args.output/arm;folder.mkdir();restore(model,panel['clean'][:2],counts[arm],'smoke');check_frozen(model,snapshots[arm])
        write_json(folder/'smoke.json',dict(status='PASS',real_L_s_IQ=2,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))
        print(json.dumps(dict(event='REAL_SOURCE_CHECKPOINT_SMOKE_PASS',arm=arm)),flush=True)
        rows[arm]=dict(parameters=d.PARAMETERS[arm],source_final=[payloads[arm]['epoch'],payloads[arm]['updates']],views={})
    for view in d.VIEWS:
        parts={a:{} for a in d.ARMS}
        for begin in range(0,256,d.BATCH):
            value=panel[view][begin:begin+d.BATCH];clean=panel['clean'][begin:begin+d.BATCH]
            outputs={a:restore(m,value,counts[a],'center') for a,m in models.items()}
            for arm in d.ARMS:
                for key,v in measure(outputs[arm],clean,outputs[d.REFERENCE_ARM]).items():parts[arm].setdefault(key,[]).append(v.detach().cpu())
        for arm in d.ARMS:
            measured=grouped({k:torch.cat(v) for k,v in parts[arm].items()});rows[arm]['views'][view]=measured
            write_json(args.output/arm/(view+'_metrics.json'),measured)
            print(json.dumps(dict(event='SOURCE_VIEW_COMPLETE',arm=arm,view=view,count=256,seconds=time.perf_counter()-started)),flush=True)
    for arm,model in models.items():
        check_frozen(model,snapshots[arm])
        if counts[arm]!=dict(smoke=2,center=768):raise ValueError('Actual source input exposure differs')
        rows[arm].update(checkpoint_unchanged=True,optimizer_updates=0,actual_IQ_inputs=counts[arm],L_s_diagnostic_inputs=384,U_s_diagnostic_inputs=384,L_s_smoke_inputs=2)
        write_json(args.output/arm/'complete.json',dict(status='SOURCE_MOMENTS_COMPLETE',arm=arm,views=list(d.VIEWS),checkpoint_unchanged=True,actual_IQ_inputs=counts[arm]))
    result=dict(status='SOURCE_MOMENTS_COMPLETE',run=d.RUN,method_commit=args.expected_commit,configuration=d.configuration(),rows=rows,seconds=time.perf_counter()-started,
        F_weight_reads=0,F_output_reads=0,TX_label_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,new_network_versions=0,all_checkpoint_states_unchanged=True)
    write_json(args.output/'results.json',result);write_json(args.output/'complete.json',dict(status=result['status'],arms=list(rows),views=list(d.VIEWS),method_commit=args.expected_commit,seconds=result['seconds'],all_checkpoint_states_unchanged=True))
    print(json.dumps(dict(event='SOURCE_MOMENTS_COMPLETE',seconds=result['seconds'])),flush=True)


def parser():
    p=argparse.ArgumentParser(description='CPU source-only fourth-order errors;no F/TX/V/target',allow_abbrev=False)
    for key in ('compact-parent','factor-run','complex-run','parent-action','source-cache','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-commit',required=True);return p


def main():
    args=parser().parse_args();args._owns_output=False;torch.set_num_threads(2);torch.set_num_interop_threads(2)
    try:diagnose(args)
    except Exception as error:
        if args._owns_output and not (args.output/'failure.json').exists():write_json(args.output/'failure.json',dict(status='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
