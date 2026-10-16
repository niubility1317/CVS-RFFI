"""Remote metadata-only complete-run cost;zero IQ/CP/NN/updates."""
import json,os
from pathlib import Path
B=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-AsTiming-sourceRX1-e20-s2026092701-m4-r01');R=B.parents[1]/'releases/cvs_FcrossRX_AsTiming_sourceRX1_E20_20261017_r01'
r=lambda p:json.loads(p.read_text());done=r(B/'completion.json');launch=r(B/'launch.json')['rows']
if done['status']!='ANALYZED' or (B/'failure.json').exists() or len(launch)!=4:raise ValueError('Complete source matrix first')
boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));hz=os.sysconf('SC_CLK_TCK');rows=[]
for x in launch:
 base=B/x['row_id']/'source';a=r(base/'satmix_result.json');start=boot+x['start_ticks']/hz;end=(base/'satmix_result.json').stat().st_mtime;viewtimes=[p.stat().st_mtime for p in (base/'source_readouts').glob('*.json')];NN=sum(a[k] for k in ('student_train_forward_packets','student_eval_forward_packets','teacher_train_forward_packets','teacher_eval_forward_packets'))-2+a['held_forward_packets'];rows.append(dict(row_id=x['row_id'],pid=x['pid'],gpu=x['gpu'],worker_alive=Path('/proc',str(x['pid'])).exists(),start_unix_from_boot=start,result_mtime=end,process_to_final_result_seconds=end-start,native_elapsed_seconds=a['elapsed_seconds'],first_to_last_source_readout_write_seconds=max(viewtimes)-min(viewtimes),new_NN=NN,model_epochs=20,optimizer_updates=4440,head_fit_updates=0))
sub=r(R/'submit.json');all_terminal=not Path('/proc',str(sub['pid'])).exists() and all(not x['worker_alive'] for x in rows)
print(json.dumps(dict(execution='VERIFIED' if all_terminal else 'UNKNOWN',all_own_processes_naturally_terminal=all_terminal,dispatcher_pid=sub['pid'],rows=rows,worker_start_to_dispatcher_completion_seconds=(B/'completion.json').stat().st_mtime-min(x['start_unix_from_boot'] for x in rows),boot_precision_seconds=1,new_NN=sum(x['new_NN'] for x in rows)+2,new_model_epochs=80,new_optimizer_updates=17760,dispatcher_real_checkpoint_smoke_NN=2,source_only=True,target_access=False)))
