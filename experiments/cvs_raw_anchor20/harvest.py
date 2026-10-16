"""Independent offline source analysis; no native, IQ or checkpoint imports."""
import argparse,csv,gzip,json,math,subprocess
from itertools import zip_longest
from pathlib import Path
from experiments.cvs_ordinary_fold20.harvest import read,write,metrics,parse_logs,SSH
ROOT=Path(__file__).resolve().parents[2]

def collect(output,spec):
    output=Path(output);(output/'logs').mkdir(parents=True,exist_ok=True)
    names=['completion.json','result.json','initialization.json','source_frozen.json','source_contract.json','actual_native_args.json',
        'module_files.json','mechanism_progress.json','epoch_metrics.jsonl','metrics_epoch.csv','step_metrics.jsonl.gz',
        'training_sequence.jsonl.gz','held_V_predictions.json','raw_anchor_proof.json','F_implementation_proof.json','F_optimizer_steps.jsonl.gz']
    for row in spec['rows']:
        source=output/row['row_id']/'source';source.mkdir(parents=True,exist_ok=True)
        files=names+(['ordinary_base_config.json'] if row['config']['arm']!='A0' else [])
        subprocess.run(['scp',*SSH,*['N607:'+row['output_root']+'/'+n for n in files],str(source)],check=True,timeout=60)
        subprocess.run(['scp',*SSH,'N607:'+row['log_path'],str(output/'logs')],check=True,timeout=60)

def analyze(output,spec,state):
    output=Path(output)
    if state['failure'] or state['dispatcher_live'] or not state['complete'] or any(r['live'] for r in state['rows']):
        raise ValueError('Terminal complete three-row source experiment required')
    results={};predictions={};curves=[];strata=[];proofs={};costs=[];args={};epochs={}
    for row in spec['rows']:
        arm=row['config']['arm'];source=output/row['row_id']/'source';v=read(source/'result.json');done=read(source/'completion.json')
        init=read(source/'initialization.json');freeze=read(source/'source_frozen.json');contract=read(source/'source_contract.json')
        f=read(source/'F_implementation_proof.json');a=read(source/'raw_anchor_proof.json')
        if (done['config']!=row['config'] or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['initialization']!='scratch'
            or done['checkpoint_sources'] or done['ancestors'] or done['head_fit_updates'] or done['target_access']
            or not init['scratch_only'] or init['checkpoint_sources'] or init['target_contact']
            or freeze['epoch']!=20 or freeze['selection']!='final_only' or freeze['target_access']
            or v['L_forward_packets']!=2273280 or v['training_loader_packets']!={'L_s':568320,'U_s':396368}
            or v['target_access'] or v['U_truth_read'] or v['head_fit_updates'] or not v['default_head_joint_training']
            or contract['held_RX']!=1 or contract['source_receivers']!=[3,4,6,8]
            or {k:len(z) for k,z in contract['role_ids'].items()}!={'L_s':5040,'U_s':45360,'V':21600}
            or f['steps']!=4440 or f['physical_clip_calls']!=4440 or f['feature']!='SIG_TIME_X3' or f['optimizer']!='CENTER_UNITS'
            or a['actual_L_forwards_per_sample']!=4 or a['pair_weights']!=[.5,.5] or a['cross_view_consistency']):
            raise ValueError('Actual F/E20/source/raw-anchor/default-head contract differs')
        if {(z['TX'],z['RX'],z['day']) for z in v['coverage']}!={(t,r,d) for t in range(6) for r in (3,4,6,8) for d in (1,2,3)} or sum(z['count'] for z in v['coverage'])!=568320:
            raise ValueError('Full identity/RX/day training coverage required')
        with gzip.open(source/'F_optimizer_steps.jsonl.gz','rt',encoding='utf-8') as stream:opt=[json.loads(z) for z in stream]
        if [z['step'] for z in opt]!=list(range(1,4441)) or any(z['physical_clip_calls']!=z['step'] or not math.isclose(z['sinc_lr'],1000*z['base_lr'],rel_tol=1e-12) or z['sinc_weight_decay']!=0 or z['physical_decay_alpha_lambda']<0 or z['sinc_eps']<=0 for z in opt):
            raise ValueError('Every physical optimizer mapping must be observed')
        mechanism=read(source/'mechanism_progress.json')['records']
        if len(mechanism)!=20 or any(z<=0 for z in v['actual_feature_parameters_moved'].values()):raise ValueError('Actual F/head movement missing')
        if arm=='A0' and v['ordinary'] is not None:raise ValueError('A0 executed unused ordinary')
        if arm!='A0' and (v['ordinary']['calls']!=4440 or v['ordinary']['packets']!=568320):raise ValueError('Exactly one G draw per L batch required')
        pred=read(source/'held_V_predictions.json')['records'];keys=[tuple(z[k] for k in ('eq','RX','day','sig_i','base_index')) for z in pred]
        if len(pred)!=5400 or len(set(keys))!=5400 or any(z['RX']!=1 for z in pred):raise ValueError('Exact held source predictions required')
        m=metrics(pred)
        if m['confusion']!=v['held_V']['confusion'] or abs(m['macro_f1']-v['held_V']['macro_f1'])>1e-12:raise ValueError('Independent head decision recount differs')
        predictions[arm]=dict(zip(keys,pred));results[arm]=v
        for dimension,values in (('TX',range(6)),('day',(1,2,3)),('TX_day',[f'{t}/{d}' for t in range(6) for d in (1,2,3)])):
            for value in values:
                selected=[z for z in pred if (z['label'] if dimension=='TX' else z['day'] if dimension=='day' else f"{z['label']}/{z['day']}")==value]
                strata.append(dict(arm=arm,dimension=dimension,value=value,**metrics(selected)))
        log=parse_logs(source,output/'logs'/(row['row_id']+'.log'));proofs[arm]=log['proof'];epochs[arm]=log['epochs']
        curves += [dict(arm=arm,**z) for z in log['curves']]
        raw=read(source/'actual_native_args.json');args[arm]=raw.get('args',raw)
        costs.append(dict(arm=arm,epochs=20,updates=4440,parent_weight_epochs=0,
            real_forward_packets=sum(v[k] for k in ('L_forward_packets','student_U_forward_packets','student_source_eval_forward_packets','EMA_eval_forward_packets','held_V_forward_packets')),
            L_forward_packets=v['L_forward_packets'],student_U_forward_packets=v['student_U_forward_packets'],EMA_forward_packets=v['EMA_eval_forward_packets'],
            source_eval_forward_packets=v['student_source_eval_forward_packets'],held_V_forward_packets=v['held_V_forward_packets'],source_seconds=v['elapsed_seconds']))
    differences={arm:sorted(k for k in args[arm] if args[arm].get(k)!=args['A0'].get(k)) for arm in ('A1','A2')}
    if any(set(z)-{'candidate_id','output_dir','use_aug'} for z in differences.values()):raise ValueError(('Unexpected actual args differences',differences))
    seq=[output/r['row_id']/'source/training_sequence.jsonl.gz' for r in spec['rows']];count=0
    with gzip.open(seq[0],'rt') as x,gzip.open(seq[1],'rt') as y,gzip.open(seq[2],'rt') as z:
        for a,b,c in zip_longest(x,y,z):
            if a!=b or a!=c:raise ValueError('All actual source sequences must match')
            count+=1
    a,b,z=(results[k] for k in ('A2','A0','A1'));h=a['held_V']
    checks=dict(vs_A0_accuracy=100*(h['accuracy']-b['held_V']['accuracy'])>=1.-1e-9,vs_A0_macroF1=h['macro_f1']>=b['held_V']['macro_f1']-1e-12,
        seen_health=100*(b['seen_V']['source_val_accuracy']-a['seen_V']['source_val_accuracy'])<=.5+1e-9,
        A1_accuracy_retained=100*(z['held_V']['accuracy']-h['accuracy'])<=.5+1e-9,A1_macroF1_retained=100*(z['held_V']['macro_f1']-h['macro_f1'])<=.5+1e-9,
        complete_health=all(v['complete_health'] for v in results.values()))
    q=state['source_decision']
    if q['checks']!=checks or q['status']!=('PASS' if all(checks.values()) else 'FAIL'):raise ValueError('Independent qualification differs')
    paired={}
    for arm in ('A0','A1'):
        if predictions[arm].keys()!=predictions['A2'].keys() or any(v['label']!=predictions[arm][k]['label'] for k,v in predictions['A2'].items()):raise ValueError('Same held physical labels required')
        paired['A2_vs_'+arm]=dict(repaired=sum(v['prediction']==v['label'] and predictions[arm][k]['prediction']!=v['label'] for k,v in predictions['A2'].items()),
            broken=sum(v['prediction']!=v['label'] and predictions[arm][k]['prediction']==v['label'] for k,v in predictions['A2'].items()))
    summary=dict(status='ANALYZED',source_decision=q,models=results,costs=costs,paired=paired,actual_native_arg_differences=differences,
        full_log_proof=proofs,actual_sequence_batch_records=count,model_epochs=60,updates=13320,
        real_forward_packets=sum(v['real_forward_packets'] for v in costs),matrix_wall_seconds=state['complete']['wall_seconds'],
        model_seed_count=1,source_fold_count=1,target_access=False,automatic_extension=False,goal_completed=False)
    for name,rows in (('training_curves.csv',curves),('source_strata.csv',strata),('measured_costs.csv',costs)):
        fields=list(dict.fromkeys(k for v in rows for k in v))
        with (output/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    write(output/'analysis_summary.json',summary);plot(output,curves,strata,epochs)
    lines=['# F 原始监督份额：RX1 三组 E20 结果','',f"源资格 **{q['status']}**。三组 scratch E20 均已完成，目标访问 0。",'',
        '|模型|held RX1 accuracy|macroF1|seen V均值|seen V最差RX|','|---|---:|---:|---:|---:|']
    for arm,v in results.items():lines.append(f"|{arm}|{100*v['held_V']['accuracy']:.6f}%|{100*v['held_V']['macro_f1']:.6f}%|{100*v['seen_V']['source_val_accuracy']:.6f}%|{100*v['seen_V']['source_val_worst_rx']:.6f}%|")
    lines+=['',f"A2−A0 accuracy {q['A2_vs_A0_accuracy_gain_pp']:+.6f}pp、macroF1 {q['A2_vs_A0_macroF1_gain_pp']:+.6f}pp；A2−A1 accuracy {q['A2_vs_A1_accuracy_gain_pp']:+.6f}pp、macroF1 {q['A2_vs_A1_macroF1_gain_pp']:+.6f}pp。",'',
        f"逐项判定：{checks}。同物理ID修复/破坏：{paired}。完整60epoch/13320step/nativeCSV/fullstdout已解析；{count}条实际L/U批次三组逐条匹配；F每行4440次物理clip/单位映射全部保存并检查。",'',
        f"新60模型轮/13320更新，真实完整网络前向{summary['real_forward_packets']}，墙钟{summary['matrix_wall_seconds']:.3f}s；无历史权重成本，本地合成检查另列。默认头/三投影/q均按原联合训练更新，未拟合额外头。",'',
        '只有一个已使用过的源RX折、一个modelseed；不等于正式clean、urban或跨seed证据。固定E20用尽不等于收敛。FAIL关闭本候选；PASS只允许另登记固定其余四RX，不能直接进入target。', '',
        '逐TX/day负结果与完整曲线均保存。图：complete_training_curves.png、held_source_strata.png。正式practical和received IQ均未改动；clean/urban双突破仍未证明。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return summary

def plot(output,curves,strata,epochs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'A0':'#386cb0','A1':'#e66101','A2':'#4c956c'}
    fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for ax,key in zip(axes.flat,('legacy_rdrop/paired_CE','legacy_rdrop/symmetric_KL','train/grad_before_clip','train/pseudo_selected','val_tx_acc','lr')):
        for arm in colors:
            rows=epochs[arm] if key in ('val_tx_acc','lr') else [v for v in curves if v['arm']==arm]
            ax.plot([v['epoch'] for v in rows],[v.get(key,0) for v in rows],label=arm,color=colors[arm])
        ax.axvline(13.5,color='grey',ls='--',lw=.8);ax.set(xlabel='Epoch',ylabel=key);ax.legend()
    fig.savefig(output/'complete_training_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,dimension in zip(axes,('TX','day')):
        for arm in colors:
            rows=[v for v in strata if v['arm']==arm and v['dimension']==dimension]
            ax.plot([v['value'] for v in rows],[100*v['accuracy'] for v in rows],marker='o',label=arm,color=colors[arm])
        ax.set(xlabel=dimension,ylabel='Held source RX1 accuracy (%)');ax.legend()
    fig.savefig(output/'held_source_strata.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    s=read(ROOT/'experiments/cvs_raw_anchor20/registration.json')
    if not a.cached:collect(a.output,s)
    v=analyze(a.output,s,read(a.state));print(json.dumps({k:z for k,z in v.items() if k not in ('models','full_log_proof')},ensure_ascii=False))
