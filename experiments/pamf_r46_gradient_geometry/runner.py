import argparse,json,os,sys,time,traceback
from pathlib import Path
import torch
from . import design as d
from .gradients import measure_batch,summarize_batches
from experiments.pamf_r46_source_response.runtime import check_frozen,load_model,validate_sources
from experiments.pamf_r46_source_fidelity.runner import load_panel
from experiments.pamf_restoration.data import write_json
from experiments.pamf_restoration.runner import code_commit


def batch_positions():
    return [list(range(i,i+8))+list(range(128+i,128+i+8)) for i in range(0,128,8)]


def diagnose(args):
    if args.output.name!=d.RUN or args.output.exists():raise ValueError('New immutable source diagnostic output required')
    if code_commit()!=args.expected_commit or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Pinned CPU-only diagnostic required')
    started=time.perf_counter();validate_sources(args)
    models={};payloads={}
    for arm in d.ARMS:models[arm],payloads[arm]=load_model(args,arm)
    panel,positions,roles=load_panel(args)
    if tuple(panel)!=d.VIEWS or len(positions)!=256 or roles.tolist()!=['L_s']*128+['U_s']*128:raise ValueError('Original fixed source-training panel required')
    args.output.mkdir();args._owns_output=True
    write_json(args.output/'launch.json',dict(status='SOURCE_GRADIENT_GEOMETRY_RUNNING',commit=args.expected_commit,pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,CPU_threads=2,GPU_workers=0,time=time.time()))
    write_json(args.output/'comparison_plan.json',d.configuration())
    write_json(args.output/'panel.json',dict(opaque_role_positions=positions.tolist(),roles=roles.tolist(),unique_source_positions=256,TX_labels_read=0,F_reads=0,V_reads=0,target_reads=0))
    snapshots={a:{k:v.detach().clone() for k,v in m.state_dict().items()} for a,m in models.items()}
    teacher=models[d.REFERENCE_ARM];teacher_inputs=0
    with torch.no_grad():
        desired_smoke=[v.detach() for v in teacher.stage_outputs(panel['received01'][:2])[0]];teacher_inputs+=2
    rows={};counts={a:dict(smoke_received=0,smoke_clean=0,received=0,clean_identity=0) for a in d.ARMS}
    for arm,model in models.items():
        folder=args.output/arm;folder.mkdir()
        probe=measure_batch(model,panel['received01'][:2],panel['clean'][:2],desired_smoke)
        counts[arm].update(smoke_received=2,smoke_clean=2);check_frozen(model,snapshots[arm])
        write_json(folder/'smoke.json',dict(status='PASS',real_L_s_physical_IQ=2,student_G_IQ_inputs=4,objectives=probe['losses'],F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,optimizer_updates=0))
        print(json.dumps(dict(event='REAL_SOURCE_GRADIENT_SMOKE_PASS',arm=arm)),flush=True)
        rows[arm]=dict(parameters=d.PARAMETERS[arm],source_final=[payloads[arm]['epoch'],payloads[arm]['updates']],views={})
    for view in d.VIEWS:
        desired=[]
        with torch.no_grad():
            for take in batch_positions():
                desired.append([v.detach() for v in teacher.stage_outputs(panel[view][take])[0]]);teacher_inputs+=16
        for arm,model in models.items():
            records=[]
            for batch,take in enumerate(batch_positions()):
                record=measure_batch(model,panel[view][take],panel['clean'][take],desired[batch])
                records.append(dict(batch=batch,count=16,L_s=8,U_s=8,**record));counts[arm]['received']+=16;counts[arm]['clean_identity']+=16
            check_frozen(model,snapshots[arm]);summary=summarize_batches(records)
            rows[arm]['views'][view]=summary
            write_json(args.output/arm/(view+'_batches.json'),dict(arm=arm,view=view,records=records))
            print(json.dumps(dict(event='SOURCE_GRADIENT_VIEW_COMPLETE',arm=arm,view=view,batches=16,unique_source_positions=256,summary_rows=len(summary),seconds=time.perf_counter()-started)),flush=True)
    if teacher_inputs!=770:raise ValueError('Fixed shared original teacher exposure differs')
    for arm,model in models.items():
        check_frozen(model,snapshots[arm])
        if counts[arm]!=dict(smoke_received=2,smoke_clean=2,received=768,clean_identity=768):raise ValueError('Actual fullG source input accounting differs')
        rows[arm].update(checkpoint_unchanged=True,optimizer_updates=0,actual_student_IQ_inputs=counts[arm],L_s_diagnostic_inputs=768,U_s_diagnostic_inputs=768,L_s_smoke_inputs=4)
        write_json(args.output/arm/'complete.json',dict(status='SOURCE_GRADIENT_GEOMETRY_COMPLETE',arm=arm,views=list(d.VIEWS),checkpoint_unchanged=True,actual_student_IQ_inputs=counts[arm]))
    summary_rows=[dict(arm=arm,source_view=view,**entry) for arm,row in rows.items() for view,entries in row['views'].items() for entry in entries]
    if len(summary_rows)!=324:raise ValueError('Complete fixed source gradient geometry matrix required')
    result=dict(status='SOURCE_GRADIENT_GEOMETRY_COMPLETE',run=d.RUN,method_commit=args.expected_commit,configuration=d.configuration(),rows=rows,summary_rows=summary_rows,
        teacher_cache_IQ_inputs=770,total_actual_G_IQ_inputs=6930,batch_records=192,seconds=time.perf_counter()-started,
        F_weight_reads=0,F_output_reads=0,TX_label_reads=0,V_reads=0,target_reads=0,optimizer_updates=0,new_network_versions=0,all_checkpoint_states_unchanged=True)
    write_json(args.output/'results.json',result)
    write_json(args.output/'complete.json',dict(status=result['status'],arms=list(rows),views=list(d.VIEWS),method_commit=args.expected_commit,seconds=result['seconds'],all_checkpoint_states_unchanged=True))
    print(json.dumps(dict(event='SOURCE_GRADIENT_GEOMETRY_COMPLETE',seconds=result['seconds'],summary_rows=324)),flush=True)


def parser():
    p=argparse.ArgumentParser(description='CPU source-only batch gradient geometry;no F/TX/V/target or optimizer',allow_abbrev=False)
    for key in ('compact-parent','factor-run','complex-run','parent-action','source-cache','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--expected-commit',required=True);return p


def main():
    args=parser().parse_args();args._owns_output=False;torch.set_num_threads(2);torch.set_num_interop_threads(2)
    try:diagnose(args)
    except Exception as error:
        if args._owns_output and not (args.output/'failure.json').exists():write_json(args.output/'failure.json',dict(status='TECHNICAL_FAILURE',error=repr(error),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
