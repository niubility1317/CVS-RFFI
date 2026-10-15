"""Offline evidence reader; never imports training code, IQ or checkpoints."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SSH=['-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def remote(script):
    result=subprocess.run(['ssh',*SSH,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+result.stdout.decode('utf-8',errors='replace'))
    return json.loads(result.stdout)


INSPECT=r'''
import json,time,subprocess
from pathlib import Path
c=CONFIG
root=Path(c['root']);release=Path(c['release'])
def read(p):
 if not p.is_file():return None
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return None
def process(pid):
 p=Path('/proc')/str(pid)
 try:
  if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':return None
  env=dict(v.split('=',1) for v in (p/'environ').read_bytes().decode().split('\0') if '=' in v)
  return dict(pid=pid,cwd=str((p/'cwd').resolve()),argv=(p/'cmdline').read_bytes().decode().split('\0'),GPU=env.get('CUDA_VISIBLE_DEVICES'))
 except (FileNotFoundError,ProcessLookupError):return None
launch=read(root/'launch.json');submit=read(release/'submit.json');rows=[]
for v in launch['rows'] if launch else []:
 source=root/v['row_id']/'source';progress=read(source/'mechanism_progress.json');log=Path(v['log']);contract=read(source/'source_contract.json')
 rows.append(dict(launch=v,live=process(v['pid']),epochs=len(progress['records']) if progress else 0,
  progress_last=progress['records'][-1] if progress else None,
  actual_args=read(source/'actual_native_args.json'),resolved=read(source/'resolved_config.json'),module_files=read(source/'module_files.json'),
  role_counts={k:len(x) for k,x in contract['role_ids'].items()} if contract else None,
  completion=read(source/'completion.json'),result=read(source/'result.json'),
  log_bytes=log.stat().st_size if log.is_file() else 0,
  noquery_smoke_PASS=('SMOKE PASS own scratch checkpoint; query_read=false' in log.read_text(errors='replace')) if log.is_file() else False,
  log_tail=log.read_text(errors='replace')[-1400:] if log.is_file() else None))
print(json.dumps(dict(read_at=time.time(),submit=submit,dispatcher_live=process(submit['pid']) if submit else None,
 preflight=read(release/'preflight.json'),queue=read(root/'queue_state.json'),rows=rows,
 complete=read(root/'complete.json'),failure=read(root/'failure.json'),source_decision=read(root/'source_decision.json'),
 matched_sequence=read(root/'matched_sequence.json'),GPU_state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True))))
'''


def inspect(output,name):
    spec=read(ROOT/'experiments/cvs_ordinary_fold20/registration.json')
    settings=dict(root=spec['execution']['remote_run_root'],release=spec['code']['cwd'])
    value=remote(INSPECT.replace('CONFIG',repr(settings)));write(Path(output)/name,value)
    print(json.dumps(dict(dispatcher_live=value['dispatcher_live'],rows=[dict(row=v['launch']['row_id'],live=v['live'],epochs=v['epochs'],
        log_bytes=v['log_bytes'],role_counts=v['role_counts'],result_status=v['result']['status'] if v['result'] else None) for v in value['rows']],
        complete=value['complete'],failure=value['failure']),ensure_ascii=False))
    return value


def metrics(records):
    if not records:raise ValueError('Empty source prediction stratum')
    cm=np.zeros((6,6),dtype=np.int64)
    for v in records:
        if v['label'] not in range(6) or v['prediction'] not in range(6):raise ValueError('Invalid default-head decision')
        cm[v['label'],v['prediction']]+=1
    n=int(cm.sum());correct=int(np.trace(cm))
    f1=2*np.diag(cm)/np.maximum(cm.sum(0)+cm.sum(1),1)
    return dict(count=n,correct=correct,accuracy=correct/n,macro_f1=float(f1.mean()),confusion=cm.tolist())


def parse_logs(source,stdout):
    epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    with (source/'metrics_epoch.csv').open(encoding='utf-8-sig',newline='') as stream:native=list(csv.DictReader(stream))
    with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:steps=[json.loads(line) for line in stream]
    if [v['epoch'] for v in epochs]!=list(range(1,21)) or len(native)!=20 or [v['step'] for v in steps]!=list(range(1,4441)):
        raise ValueError('Complete source E20/4440 structured logs required')
    critical=('train/loss','train/grad_total','train/grad_backbone','train/grad_before_clip',
        'legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','legacy_rdrop/CE1','legacy_rdrop/CE2')
    bad={k:sum(not isinstance(v['metrics'].get(k),(int,float)) or not math.isfinite(v['metrics'][k]) for v in steps) for k in critical}
    text=stdout.read_text(encoding='utf-8',errors='replace')
    errors={k:len(re.findall(k,text,re.I)) for k in ('Traceback','CUDA out of memory','FloatingPointError','Killed')}
    truth=max(v['metrics'].get('train/pseudo_truth_available',0) for v in steps)
    skipped={k:sum(v['metrics'].get(k,0) for v in steps) for k in ('train/skipped_nonfinite_loss','train/skipped_nonfinite_grad')}
    difference=max(abs(v['metrics']['train/loss']-v['metrics']['legacy_rdrop/paired_CE']-v['metrics']['legacy_rdrop/symmetric_KL']) for v in steps if v['epoch']<=13)
    if any(bad.values()) or any(errors.values()) or truth or any(skipped.values()) or difference>1e-5:
        raise ValueError(('Source full-log anomaly',bad,errors,truth,skipped,difference))
    curves=[]
    for epoch in range(1,21):
        records=[v['metrics'] for v in steps if v['epoch']==epoch]
        keys={k for r in records for k,v in r.items() if isinstance(v,(int,float)) and math.isfinite(v)}
        means={k:float(np.mean([r[k] for r in records if isinstance(r.get(k),(int,float)) and math.isfinite(r[k])])) for k in sorted(keys)}
        curves.append(dict(epoch=epoch,steps=len(records),**means))
    return dict(curves=curves,epochs=epochs,proof=dict(epochs=20,steps=4440,native_CSV_rows=len(native),
        full_stdout_bytes=stdout.stat().st_size,error_markers=errors,critical_nonfinite=bad,U_truth_available_max=truth,
        skipped=skipped,max_label_phase_objective_difference=difference,
        stdout_warning_lines=[v for v in text.splitlines() if 'Warning' in v or 'WARNING' in v]))


def transfer(output,spec):
    root=spec['execution']['remote_run_root'];output=Path(output)
    for row in spec['rows']:
        rid=row['row_id'];source=output/rid/'source';source.mkdir(parents=True,exist_ok=True)
        names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json',
            'module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz',
            'training_sequence.jsonl.gz','held_V_predictions.json']
        if row['config']['arm']=='N1':names+=['ordinary_base_config.json']
        subprocess.run(['scp',*SSH,*['N607:'+root+'/'+rid+'/source/'+v for v in names],str(source)],check=True,timeout=60)
        (output/'logs').mkdir(exist_ok=True)
        subprocess.run(['scp',*SSH,'N607:'+root+'/logs/'+rid+'.log',str(output/'logs')],check=True,timeout=60)


CHECKPOINT_AUDIT=r'''
import json,time
from pathlib import Path
c=CONFIG
root=Path(c['root']);rows=c['rows'];sources=[]
for row in rows:
 source=root/row/'source'
 def read(name):return json.loads((source/name).read_text())
 complete=read('completion.json');init=read('initialization.json');freeze=read('source_frozen.json');contract=read('source_contract.json')
 if complete['epoch']!=20 or complete['optimizer_steps']!=4440 or complete['initialization']!='scratch' or complete['checkpoint_sources'] or complete['ancestors'] or complete['target_access'] or complete['head_fit_updates']:raise ValueError('Own complete scratch source origin required')
 if not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact'] or freeze['selection']!='final_only' or freeze['target_access']:raise ValueError('Own fixed-final source provenance differs')
 if {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600} or contract['held_RX']!=1:raise ValueError('Four RX effective data contract differs')
 sources.append(source)
import torch
torch.set_num_threads(2);started=time.perf_counter();initial=[];records=[]
head='id_backbone.encoder.core.id_backbone.cls_head.'
for source in sources:
 before=torch.load(source/'initial_smoke.pt',map_location='cpu',weights_only=False)
 after=torch.load(source/'final_ssdg.pth',map_location='cpu',weights_only=False)
 if not before['scratch_only'] or after['epoch']!=20 or after['a1_ema_successful_updates']!=4440 or after['args']['baseline_ckpt'] or after['args']['teacher_ckpt'] or after['checkpoint_selection']!='final_only':raise ValueError('Own actual checkpoint origin differs')
 a,b=before['model'],after['model']
 if a.keys()!=b.keys() or tuple(b[head+'weight'].shape)!=(6,160):raise ValueError('Original joint160 default head/state layout required')
 if any(not bool(torch.isfinite(v).all()) for v in b.values() if v.is_floating_point()):raise ValueError('Nonfinite final model state')
 paths=[head+'weight',head+'gain','id_backbone.encoder.sequence_pool.local_projection.weight']
 movement={k:float((b[k]-a[k]).norm()) for k in paths}
 if any(v<=0 for v in movement.values()):raise ValueError('Default head/SIG did not jointly update')
 initial.append(a);records.append(dict(row_id=source.parent.name,state_tensors=len(a),default_head_shape=list(b[head+'weight'].shape),movement=movement))
if initial[0].keys()!=initial[1].keys() or any(not torch.equal(v,initial[1][k]) for k,v in initial[0].items()):raise ValueError('Actual native initial tensors differ between source arms')
print(json.dumps(dict(status='VERIFIED',rows=records,initial_all_tensors_equal=True,network_forwards=0,optimizer_updates=0,checkpoint_loads=4,only_own_four_RX_states=True,target_access=False,CPU_body_seconds=time.perf_counter()-started)))
'''


def checkpoint_audit(output,spec):
    settings=dict(root=spec['execution']['remote_run_root'],rows=[v['row_id'] for v in spec['rows']])
    script=CHECKPOINT_AUDIT.replace('CONFIG',repr(settings))
    command='env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 '+spec['code']['environment']+' -'
    result=subprocess.run(['ssh',*SSH,'-T','N607',command],input=script.encode('utf-8'),capture_output=True,timeout=60)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf-8',errors='replace')+result.stdout.decode('utf-8',errors='replace'))
    evidence=json.loads(result.stdout);write(Path(output)/'actual_checkpoint_state_audit.json',evidence)
    return evidence


def analyze(output,spec,state):
    output=Path(output);models={};held={};logs={};costs=[];curves=[];strata=[];arguments={}
    for registered in spec['rows']:
        rid=registered['row_id'];arm=registered['config']['arm'];source=output/rid/'source'
        result=read(source/'result.json');done=read(source/'completion.json');init=read(source/'initialization.json');freeze=read(source/'source_frozen.json')
        if (done['config']!=registered['config'] or done['optimizer_steps']!=4440 or done['epoch']!=20 or done['initialization']!='scratch'
            or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
            or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or result['target_access'] or result['U_truth_read'] or result['head_fit_updates'] or not result['default_head_joint_training']
            or result['L_forward_packets']!=1136640 or result['training_loader_packets']!={'L_s':568320,'U_s':396368}):
            raise ValueError('Fixed scratch source-fold/default-head contract differs')
        if {k:len(v) for k,v in read(source/'source_contract.json')['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}:
            raise ValueError('Actual source role counts differ')
        coverage=result['coverage']
        if sum(v['count'] for v in coverage)!=568320 or {(v['TX'],v['RX'],v['day']) for v in coverage}!={(t,r,d) for t in range(6) for r in (3,4,6,8) for d in (1,2,3)}:
            raise ValueError('Full training TX/RX/day coverage required')
        predictions=read(source/'held_V_predictions.json')['records']
        keys=[tuple(v[k] for k in ('eq','RX','day','sig_i','base_index')) for v in predictions]
        if len(predictions)!=5400 or len(set(keys))!=5400 or any(v['RX']!=1 or v['day'] not in (1,2,3) for v in predictions):
            raise ValueError('Exact unique held source V5400 required')
        recount=metrics(predictions);saved=result['held_V']
        if recount['confusion']!=saved['confusion'] or abs(recount['accuracy']-saved['accuracy'])>1e-12 or abs(recount['macro_f1']-saved['macro_f1'])>1e-12:
            raise ValueError('Independent source accuracy/F1 recount differs')
        held[arm]={k:(v['label'],v['prediction'],v['margin']) for k,v in zip(keys,predictions)}
        for dimension,values in [('day',(1,2,3)),('TX',range(6))]:
            for value in values:
                rows=[v for v in predictions if v['day' if dimension=='day' else 'label']==value]
                strata.append(dict(arm=arm,dimension=dimension,value=value,**metrics(rows)))
        for t in range(6):
            for day in (1,2,3):
                rows=[v for v in predictions if v['label']==t and v['day']==day]
                if len(rows)!=300:raise ValueError('Held V TX/day cells must contain300 packets')
                strata.append(dict(arm=arm,dimension='TX_day',value=f'{t}/{day}',**metrics(rows)))
        logs[arm]=parse_logs(source,output/'logs'/(rid+'.log'))
        curves.extend(dict(arm=arm,**v) for v in logs[arm]['curves'])
        a=read(source/'actual_native_args.json');arguments[arm]=a.get('args',a)
        mechanism=read(source/'mechanism_progress.json')['records']
        if len(mechanism)!=20 or any(v<=0 for v in result['actual_feature_parameters_moved'].values()):
            raise ValueError('Default head/SIG parameter updates or mechanism records incomplete')
        if arm=='N1' and (result['ordinary']['calls']!=4440 or result['ordinary']['packets']!=568320):
            raise ValueError('One N1 ordinary IQ generation per L step required')
        models[arm]=dict(held_V=saved,seen_V=result['seen_V'],mechanism=mechanism,coverage=coverage)
        costs.append(dict(arm=arm,epochs=20,updates=4440,parent_weight_epochs=0,L_forward_packets=1136640,
            student_U_forward_packets=result['student_U_forward_packets'],EMA_forward_packets=result['EMA_eval_forward_packets'],
            student_source_eval_forward_packets=result['student_source_eval_forward_packets'],held_V_forward_packets=5400,
            source_training_seconds=result['elapsed_seconds'],formal_forward_packets=0))
    changed=[k for k in set(arguments['N0'])|set(arguments['N1']) if arguments['N0'].get(k)!=arguments['N1'].get(k)]
    if set(changed)-{'candidate_id','output_dir','use_aug'}:raise ValueError(('Unexpected actual native args contrast',changed))
    if held['N0'].keys()!=held['N1'].keys() or any(held['N0'][k][0]!=held['N1'][k][0] for k in held['N0']):
        raise ValueError('Same held physical samples/labels required')
    repaired=broken=0
    for key,(truth,pred,margin) in held['N0'].items():
        second=held['N1'][key]
        repaired+=int(pred!=truth and second[1]==truth);broken+=int(pred==truth and second[1]!=truth)
    # Independently compare every real L/U sequence record, not just seeds.
    from itertools import zip_longest
    sequences=[output/r['row_id']/'source/training_sequence.jsonl.gz' for r in spec['rows']]
    count=0
    with gzip.open(sequences[0],'rt') as left,gzip.open(sequences[1],'rt') as right:
        for a,b in zip_longest(left,right):
            if a!=b:raise ValueError('Actual complete L/U sample sequence differs')
            count+=1
    q=state['source_decision'];a=models['N1'];b=models['N0']
    gain=100*(a['held_V']['accuracy']-b['held_V']['accuracy']);f1=a['held_V']['macro_f1']-b['held_V']['macro_f1']
    seen_drop=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy'])
    checks=dict(held_accuracy=gain>=1.-1e-9,held_macro_f1=f1>=-1e-12,seen_health=seen_drop<=.5+1e-9,full_health=True)
    if q['checks']!=checks or q['status']!=('PASS' if all(checks.values()) else 'FAIL'):
        raise ValueError('Frozen source qualification independently differs')
    summary=dict(status='ANALYZED',source_decision=q,actual_native_arg_differences=changed,
        paired_repaired=repaired,paired_broken=broken,actual_sequence_batch_records=count,
        models=models,costs=costs,full_log_proof={k:v['proof'] for k,v in logs.items()},
        epochs=40,updates=8880,real_forward_packets=sum(sum(v.get(k,0) for k in ('L_forward_packets','student_U_forward_packets','EMA_forward_packets','student_source_eval_forward_packets','held_V_forward_packets')) for v in costs),
        matrix_wall_seconds=state['complete']['wall_seconds'],target_access=False,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',
        model_seed_count=1,all_five_source_folds_complete=False,goal_completed=False)
    for name,rows in [('training_curves.csv',curves),('source_strata.csv',strata),('measured_costs.csv',costs)]:
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    plot(output,models,curves,strata,costs)
    write(output/'analysis_summary.json',summary)
    lines=['# 旧普通 IQ 增强的 RX1 留出结果','',f"源资格：**{q['status']}**。两个scratch E20已完成；目标数据访问0。",'',
        '|模型|held RX1 V准确率|macro-F1|seen RX V均值|seen最差RX|','|---|---:|---:|---:|---:|']
    for arm,v in models.items():
        lines.append(f"|{arm}|{100*v['held_V']['accuracy']:.6f}%|{100*v['held_V']['macro_f1']:.6f}%|{100*v['seen_V']['source_val_accuracy']:.6f}%|{100*v['seen_V']['source_val_worst_rx']:.6f}%|")
    lines+=['',f'N1−N0 held accuracy {gain:+.6f}pp，macroF1 {100*f1:+.6f}pp；修复{repaired}、新增错误{broken}。seen V下降{seen_drop:+.6f}pp。', '',
        f"完整解析40epoch/8880step/40行nativeCSV和两份完整stdout。实际全L/U序列{count}条逐条匹配。全部真实前向{summary['real_forward_packets']}，矩阵墙钟{summary['matrix_wall_seconds']:.3f}s；额外本地合成1536网络前向另计。",'',
        '准确率、macroF1、混淆和逐TX/day结果由保存的5400×2默认头决策独立复算；CE沿用执行器记录。L/U角色与seed相同；实际native args差异仅'+','.join(changed)+'。', '',
        '这是已使用过的sourceRX1折、一个model seed，不是目标测试或完整DG证明。U真值始终隐藏。原ordinary配方整体的阳性不等于其中PA等每项都已证明身份保持。固定预算用尽不等于收敛。', '',
        'FAIL按原规则结束本候选，不换RX/seed/参数；PASS只支持另登记其他四个源折，不自动宣称clean/urban双收益。原正式测试IQ未修改，lowurban>60%与suburban保持仍待正式证明。', '',
        '图：complete_training_curves.png、held_source_strata.png、ordinary_and_cost.png；逐数值表见同目录CSV。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary


def plot(output,models,curves,strata,costs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    fields=('legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','train/grad_before_clip','train/pseudo_selected')
    for ax,key in zip(axes.flat,fields):
        for arm in ('N0','N1'):
            rows=[v for v in curves if v['arm']==arm]
            ax.plot([v['epoch'] for v in rows],[v.get(key,0) for v in rows],label=arm)
        ax.axvline(13.5,color='grey',ls='--',lw=.8);ax.set(xlabel='Epoch',ylabel=key);ax.legend()
    fig.savefig(output/'complete_training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    for ax,dimension in zip(axes,('TX','day')):
        for arm in ('N0','N1'):
            rows=[v for v in strata if v['arm']==arm and v['dimension']==dimension]
            ax.plot([v['value'] for v in rows],[100*v['accuracy'] for v in rows],marker='o',label=arm)
        ax.set(xlabel=dimension,ylabel='Held source RX1 accuracy (%)');ax.legend()
    fig.savefig(output/'held_source_strata.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),constrained_layout=True)
    ordinary=[v['ordinary'] for v in models['N1']['mechanism']]
    axes[0].plot(range(1,21),[v['waveform_MSE'] for v in ordinary]);axes[0].set(xlabel='Epoch',ylabel='Cumulative waveform MSE',title='One ordinary generation per L step')
    keys=('L_forward_packets','student_U_forward_packets','EMA_forward_packets','student_source_eval_forward_packets','held_V_forward_packets')
    bottom=np.zeros(2)
    for key in keys:
        values=np.array([v[key] for v in costs])/1e6;axes[1].bar(['N0','N1'],values,bottom=bottom,label=key);bottom+=values
    axes[1].set(ylabel='Real packet forwards (millions)');axes[1].legend(fontsize=6)
    fig.savefig(output/'ordinary_and_cost.png',dpi=160);plt.close(fig)


def finish(output):
    output=Path(output);state=inspect(output,'terminal_state.json')
    if state['failure'] or not state['complete'] or state['complete']['status']!='ARTIFACTS_COMPLETE' or any(v['live'] for v in state['rows']):
        raise ValueError('Both fixed source rows must be terminal before completed analysis')
    if not state['matched_sequence'] or state['matched_sequence']['status']!='VERIFIED':raise ValueError('Real sample sequence proof missing')
    spec=read(ROOT/'experiments/cvs_ordinary_fold20/registration.json');transfer(output,spec)
    audit=checkpoint_audit(output,spec)
    summary=analyze(output,spec,state)
    summary['actual_checkpoint_state_audit']=audit
    write(output/'analysis_summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('models','full_log_proof')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--name',default='progress.json');parser.add_argument('--finish',action='store_true')
    args=parser.parse_args()
    if args.finish:finish(args.output)
    else:inspect(args.output,args.name)
