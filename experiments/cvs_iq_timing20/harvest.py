"""Retrieve completed source evidence and independently recount every record."""
import csv
import gzip
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
RUN='20261016-phase1-F-timing-sourceRX1-e20-s2026092701-m2-r01'
REMOTE='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN
OUT=ROOT/'local_artifacts/F_timing_preparation_r01/final'
ROWS=('B00-RX1-s2026092701','B10-RX1-s2026092701')


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def fetch():
    if OUT.exists():raise FileExistsError('Completed local evidence already exists; use cached analysis')
    script=r'''
import io,json,sys,tarfile
from pathlib import Path
root=Path(REMOTE)
if not (root/'completion.json').is_file() or (root/'failure.json').exists():raise ValueError('Source matrix incomplete or failed; do not harvest partial as final')
names=['completion.json','result.json','request.json','launch.json','queue_state.json']
for row in ROWS:
 names.append('logs/'+row+'.stdout.log')
 for name in ('completion.json','initialization.json','source_contract.json','source_frozen.json','resolved_config.json','actual_native_args.json','actual_modules.json','source_fold.json','timing_result.json','timing_steps.jsonl.gz','unit_steps.jsonl.gz','training_sequence.jsonl.gz','epoch_metrics.jsonl','epoch_metrics.csv','metrics_epoch.csv','metrics_epoch.jsonl','step_metrics.jsonl.gz','source_readouts/FINAL.json'):
  names.append(row+'/source/'+name)
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as t:
 for name in names:
  p=root/name
  if not p.is_file():raise FileNotFoundError(p)
  t.add(p,arcname=name,recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''.replace('REMOTE',repr(REMOTE)).replace('ROWS',repr(ROWS))
    args=['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','-T','N607','/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -']
    result=subprocess.run(args,input=script.encode(),capture_output=True)
    if result.returncode:raise RuntimeError(result.stderr.decode('utf8',errors='replace'))
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as t:
        for member in t.getmembers():
            if not member.isfile() or not (OUT/member.name).resolve().is_relative_to(OUT.resolve()):raise ValueError('Invalid local evidence path')
        OUT.mkdir(parents=True);t.extractall(OUT)


def score(labels,predictions):
    if len(labels)!=len(predictions) or any(type(v) is not int or v not in range(6) for v in labels+predictions):raise ValueError('Legal complete source decisions required')
    cm=[[0]*6 for _ in range(6)]
    for y,p in zip(labels,predictions):cm[y][p]+=1
    f1=[]
    for c in range(6):
        tp=cm[c][c];fp=sum(row[c] for row in cm)-tp;fn=sum(cm[c])-tp
        f1.append(2*tp/max(1,2*tp+fp+fn))
    return dict(count=len(labels),accuracy=sum(cm[i][i] for i in range(6))/len(labels),macro_F1=sum(f1)/6,confusion=cm)


def verify_native_epochs():
    result={}
    for row in ROWS:
        source=OUT/row/'source'
        native=[json.loads(line) for line in (source/'metrics_epoch.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
        compact=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
        with (source/'metrics_epoch.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
        expected=list(range(1,21))
        if [v['epoch'] for v in native]!=expected or [int(v['epoch']) for v in table]!=expected or len(compact)!=20:raise ValueError('Native CSV/JSONL/compact epoch coverage differs')
        keys=('train_loss_tx_labeled','train_loss_unlabeled','val_tx_acc','lr')
        for a,b,c in zip(native,compact,table):
            if any(abs(float(a[k])-float(b[k]))>1e-10 or abs(float(a[k])-float(c[k]))>1e-10 for k in keys):raise ValueError('Native epoch source differs from complete plotted curve')
        result[row]=dict(native_JSONL_records=len(native),native_CSV_records=len(table),compact_records=len(compact),key_curve_fields_exact=True)
    return result


def analyze():
    complete=read(OUT/'completion.json');results={};curves=[];decisions={};sequence_reference=None;execution={};input_audit={};native_args={}
    for row in ROWS:
        source=OUT/row/'source';result=read(source/'timing_result.json');decision=read(source/'source_readouts/FINAL.json')
        s=score(decision['labels'],decision['predictions'])
        if s['count']!=1260 or abs(s['accuracy']-result['held_default_head']['accuracy'])>1e-12 or abs(s['macro_F1']-result['macro_F1'])>1e-12:raise ValueError('Independent default-head source score differs')
        if len({tuple(p) for p in decision['physical_ids']})!=1260 or {p[1] for p in decision['physical_ids']}!={1}:raise ValueError('Held source IDs differ')
        epochs=[json.loads(line) for line in (source/'epoch_metrics.jsonl').read_text(encoding='utf8').splitlines()]
        if len(epochs)!=20:raise ValueError('Complete twenty-epoch logs required')
        curves.extend(dict(row=row,**v) for v in epochs)
        steps=[];loss_max=0.;truth=0;skips=0;aggregate={}
        with gzip.open(source/'step_metrics.jsonl.gz','rt',encoding='utf8') as f:
            for line in f:
                record=json.loads(line);steps.append(record['step']);m=record['metrics']
                raw=m['timing/raw_CE']+m['timing/raw_KL'];g=m['timing/g_slot_CE']+m['timing/g_slot_KL'];s_loss=m['timing/s_slot_CE']+m['timing/s_slot_KL']
                loss_max=max(loss_max,abs(.5*raw+.25*g+.25*s_loss-m['timing/total_L']))
                truth+=int(m.get('train/pseudo_truth_available',0) or 0);skips+=int(m.get('train/skipped_nonfinite_loss',0) or m.get('train/skipped_nonfinite_grad',0) or 0)
                epoch=(record['step']-1)//222+1
                target=aggregate.setdefault(epoch,{})
                for key in ('timing/total_L','timing/raw_CE','timing/g_slot_CE','timing/s_slot_CE','timing/raw_KL','timing/g_slot_KL','timing/s_slot_KL','timing/waveform_MSE','train/loss_unlabeled','train/pseudo_selected'):
                    value=m.get(key)
                    if isinstance(value,(int,float)):target[key]=target.get(key,0.)+value/222
        if steps!=list(range(1,4441)) or truth or skips or loss_max>1e-5:raise ValueError('Complete finite legal source trajectory/loss reconstruction failed')
        for record in curves:
            if record['row']==row:record.update(aggregate[record['epoch']])
        for name in ('timing_steps.jsonl.gz','unit_steps.jsonl.gz'):
            with gzip.open(source/name,'rt',encoding='utf8') as f:records=[json.loads(line) for line in f]
            if len(records)!=4440 or [v['step'] for v in records]!=steps:raise ValueError('Missing actual timing/unit steps')
            if name=='timing_steps.jsonl.gz':
                input_audit[row]=dict(mean_waveform_MSE=sum(v['waveform_MSE'] for v in records)/4440,max_relative_RMS_power_error=max(v['max_relative_power_error'] for v in records),max_abs_delay_samples=max(v['max_abs_delay'] for v in records))
        native_args[row]=read(source/'actual_native_args.json')
        with gzip.open(source/'training_sequence.jsonl.gz','rt',encoding='utf8') as f:sequence=[json.loads(line) for line in f]
        if any(p[1]==1 for batch in sequence for p in batch['ids']):raise ValueError('Held RX appeared in training')
        if sequence_reference is None:sequence_reference=sequence
        elif sequence!=sequence_reference:raise ValueError('Actual paired L/U physical sequences differ')
        observed={role:sum(len(v['ids']) for v in sequence if v['role']==role) for role in ('L','U')}
        unique={role:len({tuple(p) for v in sequence if v['role']==role for p in v['ids']}) for role in ('L','U')}
        if observed!={'L':568320,'U':396368} or unique!={'L':5040,'U':45360}:raise ValueError('Complete actual L/U coverage differs')
        stdout=(OUT/'logs'/(row+'.stdout.log')).read_text(encoding='utf8',errors='replace')
        fatal=[v for v in stdout.splitlines() if any(marker in v for marker in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','SIGKILL'))]
        if fatal:raise ValueError('Training stdout technical failures: '+str(fatal))
        if 'SMOKE PASS own scratch checkpoint; query_read=false' not in stdout:raise ValueError('Missing own initial real-checkpoint smoke')
        # The EMA is deepcopied after the student's 2-packet initial smoke.
        # Its inherited eval counter2 is not an executed teacher forward.
        actual_total=result['student_train_forward_packets']+result['student_eval_forward_packets']+result['teacher_train_forward_packets']+result['teacher_eval_forward_packets']-2+result['held_forward_packets']
        execution[row]=dict(full_steps=len(steps),full_epochs=len(epochs),max_weighted_L_loss_reconstruction_error=loss_max,U_truth_records=truth,nonfinite_skips=skips,training_sequence_batches=len(sequence),L_observation_batches=sum(v['role']=='L' for v in sequence),U_observation_batches=sum(v['role']=='U' for v in sequence),physical_unique=unique,physical_observations=observed,stdout_lines=len(stdout.splitlines()),stdout_warnings=[v for v in stdout.splitlines() if 'warning' in v.lower()],total_actual_network_forward_packets=actual_total,counter_note='student counts include initialsmoke2; teacher inherited counter2 subtracted; held1260 added; no extra smoke2 double count',elapsed_seconds=result['elapsed_seconds'])
        s.update(arm=result['arm'],seen_V=result['seen_V'],mean_CE=result['held_default_head']['mean_CE'],final_L_CE=result['final_L_CE'],L_forward_packets=result['L_forward_packets'],actual_steps=4440,head_fit_updates=0,target_access=False)
        results[result['arm']]=s;decisions[result['arm']]=decision
    b,g=results['B00'],results['B10'];acc=100*(g['accuracy']-b['accuracy']);f1=100*(g['macro_F1']-b['macro_F1']);drop=100*(b['seen_V']['source_val_accuracy']-g['seen_V']['source_val_accuracy'])
    conditions=dict(healthy=all(v['final_L_CE']<=.5 and v['L_forward_packets']==3409920 for v in results.values()),held_accuracy=acc>=1.,held_macroF1=f1>=0.,seen_V=drop<=.5)
    qualification=dict(qualified=all(conditions.values()),conditions=conditions,accuracy_gain_pp=acc,macroF1_gain_pp=f1,seen_V_drop_pp=drop)
    if qualification['qualified']!=complete['qualified']:raise ValueError('Independent fixed source decision differs')
    before,after=(decisions[k] for k in ('B00','B10'))
    if before['labels']!=after['labels'] or before['physical_ids']!=after['physical_ids']:raise ValueError('Source pairing mismatch')
    pairs=dict(repaired=sum(b!=y and g==y for y,b,g in zip(before['labels'],before['predictions'],after['predictions'])),broken=sum(b==y and g!=y for y,b,g in zip(before['labels'],before['predictions'],after['predictions'])))
    strata=[]
    for dimension in ('TX','day'):
        for value in (range(6) if dimension=='TX' else (1,2,3)):
            mask=[i for i,(y,p) in enumerate(zip(before['labels'],before['physical_ids'])) if (y if dimension=='TX' else p[2])==value]
            values={arm:score([decisions[arm]['labels'][i] for i in mask],[decisions[arm]['predictions'][i] for i in mask]) for arm in ('B00','B10')}
            strata.append(dict(dimension=dimension,value=value,count=len(mask),B00_accuracy=values['B00']['accuracy'],B10_accuracy=values['B10']['accuracy'],gain_pp=100*(values['B10']['accuracy']-values['B00']['accuracy'])))
    differences=[k for k,v in native_args[ROWS[0]].items() if v!=native_args[ROWS[1]].get(k)]
    if set(differences)!={'candidate_id','output_dir'}:raise ValueError('Matched actual native arguments differ scientifically')
    summary=dict(status='ANALYZED',run_id=RUN,actual_execution='VERIFIED',qualification=qualification,results=results,paired=pairs,execution=execution,input_audit=input_audit,native_epochs=verify_native_epochs(),native_argument_differences=differences,all_L_U_sequences_identical=True,head_fit_updates=0,target_access=False,new_model_epochs=40,new_optimizer_updates=8880,scope='One model seed, heldsource RX1 development; no formal clean/practical or crossseed claim')
    dump(OUT/'analysis_summary.json',summary)
    for name,records in (('training_curves.csv',curves),('source_strata.csv',strata)):
        fields=sorted({key for r in records for key in r})
        with (OUT/name).open('w',encoding='utf8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--cached',action='store_true');args=parser.parse_args()
    if not args.cached:fetch()
    analyze()
