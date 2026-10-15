"""Completed-run analysis only; no model/IQ/target-truth imports or inputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path,PurePosixPath
import subprocess
import numpy as np
from experiments.cvs_mixed_iq20.analyze import read,write,table,parse_source,source_groups,merge,VIEWS,SCENES
from experiments.cvs_urban_sigtime.operations import remote,SSH

ROOT=Path(__file__).resolve().parents[2]
SPEC=ROOT/'experiments/cvs_source_response20/train_registration.json'


def source_qualification(value):
    if value['status']!='SOURCE_QUALIFICATION_FROZEN' or value['target_scores_read'] or set(value['rows'])!={'EMP','ROT'}:
        raise ValueError('Both source-only decisions must precede result analysis')
    for decision in value['rows'].values():
        if decision['status']!='SOURCE_DECISION_FROZEN' or decision['target_metrics_read'] or not decision['before_own_target_prediction']:
            raise ValueError('Own target prediction must follow frozen source decision')
        a=decision['source_metrics'];b=decision['control']['source_metrics']
        checks=dict(clean_gain=a['source_val_accuracy']>=b['source_val_accuracy']+.005,
            worst_RX_health=a['source_val_worst_rx']>=b['source_val_worst_rx']-.01)
        if checks!=decision['checks'] or all(checks.values())!=decision['passed']:
            raise ValueError('Fixed source decision arithmetic differs')
    if value['passed']!=value['rows']['EMP']['passed']:raise ValueError('EMP must remain the fixed primary')
    delta=value['rows']['EMP']['source_metrics']['source_val_accuracy']-value['rows']['ROT']['source_metrics']['source_val_accuracy']
    if abs(100*delta-value['EMP_minus_ROT_source_clean_pp'])>1e-10:raise ValueError('Source direction comparison arithmetic differs')


def state_script(root):
    return r'''
import json,time
from pathlib import Path
root=Path(ROOT)
def read(p):
 if not p.is_file():return None
 try:return json.loads(p.read_text())
 except json.JSONDecodeError:return None
complete=read(root/'complete.json');failure=read(root/'failure.json');launch=read(root/'launch.json');dispatch=read(root/'dispatcher.json')
live=[]
for pid in [*(([dispatch['pid']]) if dispatch else []),*([v['pid'] for v in launch['rows']] if launch else [])]:
 p=Path('/proc')/str(pid)
 try:
  if p.exists() and not any(v.startswith('State:') and 'Z' in v for v in (p/'status').read_text().splitlines()):live.append(pid)
 except FileNotFoundError:pass
q=read(root/'source_qualification.json') if complete else None
clocks=None
if complete:
 clocks=dict(matrix_start=(root/'dispatcher.json').stat().st_mtime,matrix_end=(root/'complete.json').stat().st_mtime,
  row_end={v['row_id']:(root/v['row_id']/'complete.json').stat().st_mtime for v in launch['rows']})
print(json.dumps(dict(read_at=time.time(),complete=complete,failure=failure,live_pids=live,
 queue=read(root/'queue_state.json'),qualification=q,clocks=clocks)))
'''.replace('ROOT',repr(root))


def ready(state):
    done=state.get('complete')
    if done is None:return False
    if state.get('failure') or done.get('status')!='ARTIFACTS_COMPLETE' or done.get('actual_model_count')!=2 or done.get('seed_count')!=1 or not done.get('truth_last'):
        raise ValueError('Need complete fixed two-row truth-last matrix')
    source_qualification(state['qualification']);return True


def transfer(source,relative,output):
    target=output/relative;target.mkdir(parents=True,exist_ok=True)
    names=source[1]
    subprocess.run(['scp',*SSH,*['N607:'+source[0]+'/'+v for v in names],str(target)],check=True,timeout=60)


def scores(row,arm,all_records):
    scored=read(row/'scoring/complete.json');pred=read(row/'prediction/complete.json');saved=read(row/'scoring/scores.json')
    if (scored['status']!='SCORED_COMPLETE' or tuple(scored['views'])!=VIEWS or scored['predictions']!=1176000
        or scored['metric_records']!=126 or not scored['truth_last'] or pred['truth_read'] or pred['query_fit']
        or not pred['all_parameters_frozen'] or pred['head_fit_updates'] or not saved['truth_last']
        or pred['count']!=168000 or tuple(pred['views'])!=VIEWS):
        raise ValueError('Need fixed complete seven views and independent default-head scores')
    records=saved['results'];overall={v['view']:v for v in records if v['dimension']=='overall'}
    if len(records)!=126 or set(overall)!=set(VIEWS):raise ValueError('Incomplete formal score strata')
    matrix=[]
    for view in VIEWS:
        rs=[v['accuracy'] for v in records if v['view']==view and v['dimension']=='receiver']
        if len(rs)!=7:raise ValueError('All fixed formal RX required')
        matrix.append(dict(arm=arm,seed=2026092701,view=view,accuracy_pct=100*overall[view]['accuracy'],
            macro_F1_pct=100*overall[view]['macro_f1'],worst_RX_pct=100*min(rs),
            clean_minus_view_pp=100*(overall['clean']['accuracy']-overall[view]['accuracy'])))
    all_records.extend(dict(v,analysis_arm=arm) for v in records)
    return matrix


def validate_row(row,c,qualification):
    source=row/'source';done=read(source/'completion.json');init=read(source/'initialization.json')
    mechanism=read(source/'training_mechanism_complete.json');coverage=read(source/'response_coverage_complete.json')
    if (done['config']!=c or done['epoch']!=20 or done['optimizer_steps']!=4440 or done['logged_steps']!=4440
        or done['target_access'] or done['head_fit_updates'] or done['checkpoint_sources'] or done['ancestors']
        or not init['scratch_only'] or init['target_contact'] or mechanism['L_forward_packets']!=3409920
        or mechanism['U_truth_read'] or mechanism['head_fit_updates'] or mechanism['new_U_augmentation']
        or not mechanism['default_head_joint_training'] or mechanism['satellite_packets']!=0):
        raise ValueError('Own scratch E20/default-head/L-only Ag contract differs')
    decision=read(source/'source_decision.json')
    if decision!=qualification['rows'][c['arm']]:raise ValueError('Own source decision differs from frozen aggregate')
    if read(row/'prediction/complete.json')['config']!=c:raise ValueError('Formal prediction checkpoint config differs')
    sessions={};cells=set();total=0
    for value in coverage['records']:
        key=tuple(value[k] for k in ('epoch','RX','day','eq','view'));parameters=(value['bank_index'],value['strength'])
        if key in sessions and sessions[key]!=parameters:raise ValueError('Session parameters depend on TX')
        sessions[key]=parameters;cells.add((*key,value['TX']));total+=value['count']
    expected={(e,rx,day,1,0) for e in range(1,21) for rx in (1,3,4,6,8) for day in (1,2,3)}
    if (total!=568320 or set(sessions)!=expected or cells!={(*v,tx) for v in expected for tx in range(6)}
        or coverage['TX_used_for_parameters'] or coverage['packets']!=total):raise ValueError('Incomplete actual G coverage')
    for value in read(source/'source_panel_results.json')['views'].values():
        if value['count']!=720 or abs(merge([value])['accuracy']-value['accuracy'])>1e-12:raise ValueError('Source panel recount differs')
    return mechanism,coverage,done


def analyze(output,spec,state):
    q=state['qualification'];source_qualification(q);all_scores=[];matrix=[];costs=[];logs={};panels={};coverage_rows=[]
    launches={v['row_id']:v for v in read(output/'launch.json')['rows']}
    for registered in spec['rows']:
        c=registered['config'];arm=c['arm'];row=output/c['row_id']
        mechanism,coverage,done=validate_row(row,c,q);logs[arm]=parse_source(row)
        proof=logs[arm]['proof']
        if proof['skipped_nonfinite_loss'] or proof['skipped_nonfinite_grad']:raise ValueError('Skipped optimizer updates')
        for curve in logs[arm]['curves']:
            if curve['mixedIQ/g_waveform_MSE']<=0 or curve['mixedIQ/g_power_error']>2e-6:raise ValueError('Actual G absent or lost RMS')
        panels[arm]=read(row/'source/source_panel_results.json');matrix.extend(scores(row,arm,all_scores))
        prediction=read(row/'prediction/complete.json')
        costs.append(dict(arm=arm,new_model=True,epochs=20,updates=4440,source_seconds=done['elapsed_seconds'],
            row_wall_seconds=state['clocks']['row_end'][c['row_id']]-launches[c['row_id']]['launched_at'],
            L_forward_packets=mechanism['L_forward_packets'],student_U_forward_packets=mechanism['native_student_U_forward_packets'],
            EMA_forward_packets=mechanism['EMA_eval_forward_packets'],student_source_eval_forward_packets=mechanism['student_eval_forward_packets'],
            source_panel_forward_packets=5040,formal_forward_packets=prediction['actual_forward_packets']))
        coverage_rows.extend(dict(arm=arm,**v) for v in coverage['records'])
    for arm in ('B00','R_LEGACY'):
        matrix.extend(scores(output/'controls'/arm,arm,all_scores))
    index={(v['arm'],v['view']):v for v in matrix};paired=[]
    for a,b in (('EMP','B00'),('ROT','B00'),('EMP','ROT'),('EMP','R_LEGACY')):
        for view in VIEWS:paired.append(dict(comparison=a+'-'+b,view=view,
            accuracy_delta_pp=index[a,view]['accuracy_pct']-index[b,view]['accuracy_pct'],
            macro_F1_delta_pp=index[a,view]['macro_F1_pct']-index[b,view]['macro_F1_pct']))
    aggregate=[]
    for arm in ('B00','R_LEGACY','EMP','ROT'):
        rx=defaultdict(list)
        for v in all_scores:
            if v['analysis_arm']==arm and v['view'] in SCENES[3:] and v['dimension']=='receiver':rx[v['stratum']].append(v['accuracy'])
        if len(rx)!=7 or any(len(v)!=3 for v in rx.values()):raise ValueError('Complete urban RX aggregates required')
        aggregate.append(dict(arm=arm,clean_pct=index[arm,'clean']['accuracy_pct'],low_urban_pct=index[arm,SCENES[-1]]['accuracy_pct'],
            mid_urban_pct=index[arm,SCENES[-2]]['accuracy_pct'],urban_mean_pct=sum(index[arm,s]['accuracy_pct'] for s in SCENES[3:])/3,
            urban_macro_F1_pct=sum(index[arm,s]['macro_F1_pct'] for s in SCENES[3:])/3,worst_urban_RX_pct=100*min(sum(v)/3 for v in rx.values()),
            suburban_mean_pct=sum(index[arm,s]['accuracy_pct'] for s in SCENES[:3])/3))
    states=[dict(arm=arm,**v) for arm,panel in panels.items() for v in source_groups(panel)]
    curves=[dict(arm=arm,**v) for arm,log in logs.items() for v in log['curves']]
    totals={k:sum(v[k] for v in costs) for k in ('epochs','updates','L_forward_packets','student_U_forward_packets','EMA_forward_packets',
        'student_source_eval_forward_packets','source_panel_forward_packets','formal_forward_packets')}
    totals['shared_source_preparation_forward_packets']=722
    totals['matrix_wall_seconds']=state['clocks']['matrix_end']-state['clocks']['matrix_start']
    totals['all_new_forward_packets']=sum(totals[k] for k in ('L_forward_packets','student_U_forward_packets','EMA_forward_packets',
        'student_source_eval_forward_packets','source_panel_forward_packets','formal_forward_packets'))+722
    result=dict(status='ANALYZED',run_id=spec['run_id'],seed=2026092701,new_model_count=2,reused_control_count=2,
        source_qualification=q,formal_matrix=matrix,formal_aggregate=aggregate,paired=paired,costs=costs,totals=totals,
        full_log_proof={k:v['proof'] for k,v in logs.items()},head_fit_updates=0,truth_last=True,target_feedback=False,
        goal_completed=False,cross_seed_reproducibility=False,training_As=False,
        scope='Independent Ag only; no current joint row, no interaction estimate or low>60 claim')
    result['paired_G_coverage_metadata_equal']=(read(output/'EMP-s2026092701/source/response_coverage_complete.json')['records']==
        read(output/'ROT-s2026092701/source/response_coverage_complete.json')['records'])
    for name,values in [('seven_view_results.csv',matrix),('paired_effects.csv',paired),('measured_costs.csv',costs),
        ('complete_training_curves.csv',curves),('source_state_quality.csv',states),('actual_G_coverage.csv',coverage_rows),('formal_strata.csv',all_scores),
        ('formal_aggregate.csv',aggregate)]:table(output/name,values)
    figures(output,matrix,paired,logs,costs,states,coverage_rows)
    write_report(output,result,panels)
    write(output/'analysis_summary.json',result)
    return result


def figures(output,matrix,paired,logs,costs,states,coverage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm in ('R_LEGACY','B00','EMP','ROT'):
        lookup={v['view']:v['accuracy_pct'] for v in matrix if v['arm']==arm}
        axes[0].plot(range(7),[lookup[v] for v in VIEWS],marker='o',label=arm)
    axes[0].set(xticks=range(7),xticklabels=['Clean','HighS','MidS','LowS','HighU','MidU','LowU'],ylabel='Accuracy (%)',title='Fixed seven-view default-head results');axes[0].legend()
    for name in ('EMP-B00','ROT-B00','EMP-ROT'):
        lookup={v['view']:v['accuracy_delta_pp'] for v in paired if v['comparison']==name}
        axes[1].plot(range(7),[lookup[v] for v in VIEWS],marker='o',label=name)
    axes[1].axhline(0,color='gray',lw=1);axes[1].set(xticks=range(7),xticklabels=['Clean','HighS','MidS','LowS','HighU','MidU','LowU'],ylabel='Paired difference (pp)',title='One model seed; no joint row');axes[1].legend()
    fig.savefig(output/'formal_results.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for arm,value in logs.items():
        curves=value['curves'];epoch=[v['epoch'] for v in curves]
        for ax,key,title in ((axes[0,0],'mixedIQ/mixed_L_loss','L objective'),(axes[0,1],'train/grad_total','Total gradient norm'),
            (axes[1,0],'mixedIQ/g_waveform_MSE','Actual G waveform MSE')):
            ax.plot(epoch,[v[key] for v in curves],label=arm);ax.set(xlabel='Epoch',title=title)
        axes[1,1].plot(epoch,[v['val_tx_acc'] for v in value['epochs']],label=arm)
    axes[1,1].set(xlabel='Epoch',title='Source V accuracy (native telemetry)')
    for ax in axes.flat:ax.legend()
    for ax in axes.flat:ax.axvline(13.5,color='gray',ls=':',lw=.8)
    fig.savefig(output/'complete_training_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for arm in ('EMP','ROT'):
        values=[v for v in states if v['arm']==arm and v['view']=='practical_low_urban' and v['dimension']=='lock']
        axes[0].plot([v['stratum'] for v in values],[100*v['accuracy'] for v in values],marker='o',label=arm)
        values=[sum(v['count'] for v in coverage if v['arm']==arm and v['epoch']==e) for e in range(1,21)]
        axes[1].plot(range(1,21),values,label=arm)
    axes[0].set(ylabel='Source panel accuracy (%)',title='Frozen source low-urban lock strata');axes[0].legend()
    axes[1].set(xlabel='Epoch',ylabel='Actual G packets',title='G coverage; every session covers all six TX');axes[1].legend()
    fig.savefig(output/'source_strata_and_coverage.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for arm in ('EMP','ROT'):
        values=[v for v in states if v['arm']==arm and v['view']=='practical_low_urban' and v['dimension']=='state']
        axes[0].plot([v['stratum'] for v in values],[100*v['accuracy'] for v in values],marker='o',label=arm)
        values=[v for v in states if v['arm']==arm and v['view']=='practical_low_urban' and v['dimension']=='added_noise_SNR_bin']
        bins=['<0','0-3','3-10','10-20','>=20'];lookup={v['stratum']:v for v in values}
        axes[1].plot(bins,[100*lookup[k]['accuracy'] if k in lookup else np.nan for k in bins],marker='o',label=arm)
        progress=read(output/(arm+'-s2026092701')/'source/training_mechanism_progress.json')['records']
        axes[2].plot([v['epoch'] for v in progress],[np.mean(v['gates']['signature_gain']) for v in progress],label=arm)
    axes[0].set(title='Source low urban: propagation state',ylabel='Accuracy (%)');axes[1].set(title='Source low urban: added-noise bins',xlabel='Signal / added noise (dB)',ylabel='Accuracy (%)')
    axes[2].set(title='Existing SIG_TIME internal gain',xlabel='Epoch',ylabel='Mean bounded gain')
    for ax in axes:ax.legend()
    fig.savefig(output/'source_quality_and_existing_gate.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].bar([v['arm'] for v in costs],[v['source_seconds'] for v in costs]);axes[0].set(title='Measured source training',ylabel='Wall time (seconds)')
    names=['L','Student U','EMA','Source eval','Source panel','Formal']
    for v in costs:axes[1].plot(range(6),[v[k]/1e6 for k in ('L_forward_packets','student_U_forward_packets','EMA_forward_packets',
        'student_source_eval_forward_packets','source_panel_forward_packets','formal_forward_packets')],marker='o',label=v['arm'])
    axes[1].set(xticks=range(6),xticklabels=names,title='Actual forward packets',ylabel='Millions');axes[1].legend()
    fig.savefig(output/'measured_cost.png',dpi=180);plt.close(fig)


def write_report(output,result,panels):
    q=result['source_qualification'];totals=result['totals']
    lines=['# 源响应方向的独立 Ag：E20 完整结果','',
        '两个新 scratch 模型，固定 seed2026092701，fullL6300/U56700/V27000、E20=13L+7U/4440更新。SIG_TIME/RAW_HZ/NoExplicitCFO、joint160默认cosine头与原U/ownEMA联合训练。只替换G，raw/G/raw权重.25/.25/.50，每槽同IQ两次RDrop；不训练As。B00是同六前向匹配控制，R_LEGACY是原普通增强参照，均只读复用原产物，不算新模型或新seed。',
        '', '本轮不含B11，因此没有联合收益或交互项；不把独立Ag的变化冒充已经完成双增强目标。原七视图预测固定后由独立CUDA隐藏CPU truth-last评分，目标成绩不回流选择。',
        '', '## 源域决定先固定','',f"EMP资格：{'PASS' if q['passed'] else 'FAIL'}。固定要求fullV相对B00至少+0.5pp、最差源RX下降≤1pp。",
        '', '|行|fullV准确率 %|相对B00 pp|最差源RX %|资格|','|---|---:|---:|---:|---|']
    for arm,v in q['rows'].items():
        m=v['source_metrics'];lines.append(f"|{arm}|{100*m['source_val_accuracy']:.6f}|{v['clean_gain_pp']:+.6f}|{100*m['source_val_worst_rx']:.6f}|{'PASS' if v['passed'] else 'FAIL'}|")
    lines+=['',f"EMP−ROT源clean为{q['EMP_minus_ROT_source_clean_pp']:+.6f}pp。两行source_decision均在自身target预测前冻结，完整判断只读取source文件。V中的RX已见于源训练，不称为未见RX泛化。",'',
        '## 正式默认头结果','', '|行|clean %|low urban %|mid urban %|urban均值 %|urban宏F1 %|最差urban RX %|suburban均值 %|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in result['formal_aggregate']:
        lines.append('|'+v['arm']+'|'+'|'.join(f'{v[k]:.6f}' for k in ('clean_pct','low_urban_pct','mid_urban_pct','urban_mean_pct','urban_macro_F1_pct','worst_urban_RX_pct','suburban_mean_pct'))+'|')
    lines+=['','|行|视图|准确率 %|宏F1 %|最差RX %|clean−本视图 pp|','|---|---|---:|---:|---:|---:|']
    for v in result['formal_matrix']:lines.append(f"|{v['arm']}|{v['view']}|{v['accuracy_pct']:.6f}|{v['macro_F1_pct']:.6f}|{v['worst_RX_pct']:.6f}|{v['clean_minus_view_pp']:.6f}|")
    lines+=['','全部28行×18分层评分记录见formal_strata.csv，其中两条新模型252条，两个旧控制252条。单模型seed，不提供跨seed复现或独立盲测声明。每个新模型7×168000=1176000个正式决定，原IQ、SNR、场景和困难样本不变。',
        '', '## 完整训练与机制','',f"完整解析{totals['epochs']}个epoch/{totals['updates']}个step；两个新行的关键loss/梯度有限、跳过更新和U真值标记均为0。每行568320个G包，300个epoch×RX/day session、1800个session×TX单元全部覆盖，参数在session内跨TX一致。",'',
        '源冻结panel720复用固定父模型clean正确集合，head margin统一除以30；各状态、lock、signal_to_added_noise区间准确率/F1/保持率/间隔见source_state_quality.csv。新增噪声SNR不是总SNR。这里没有As训练，源panel中的传播状态结果只能描述Ag模型对既有损伤的表现，不证明噪声/多径的单独因果改善。']
    for arm,panel in panels.items():
        v=panel['views']['practical_low_urban'];lines.append(f"{arm}源low：{v['correct']}/{v['count']}，默认头间隔/30={v['margin']:.9f}，固定父clean正确集合保持率={100*v['clean_correct_retention']:.6f}%。")
    lines+=['','## 实测成本','', '|行|source秒|row墙钟秒|L前向|studentU前向|EMA前向|source eval前向|panel前向|formal前向|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for v in result['costs']:lines.append(f"|{v['arm']}|{v['source_seconds']:.6f}|{v['row_wall_seconds']:.6f}|"+'|'.join(str(v[k]) for k in ('L_forward_packets','student_U_forward_packets','EMA_forward_packets','student_source_eval_forward_packets','source_panel_forward_packets','formal_forward_packets'))+'|')
    lines+=['',f"两新模型共{totals['epochs']}model-epoch/{totals['updates']}更新；本轮源/测试真实数据新增前向{totals['all_new_forward_packets']}（含共享source smoke2+父clean panel720）。矩阵墙钟{totals['matrix_wall_seconds']:.6f}秒。完成E20表示预算用尽，不等于收敛。",'',
        'B00/R_LEGACY的既有训练、父只读E20参考的20epoch/4440更新/923.560061秒、先前有限IQ安全probe的189002前向/120.415114秒分别保留，不混作本轮增量训练或继承成本。源panel构造还包含4320次物理信道处理，属于IQ生成而非网络前向。',
        '', '## 判定与交付','',
        '源规则FAIL：结束本固定候选，不开展其联合训练或种子扩展。' if not q['passed'] else '源规则PASS：仅支持另行预登记的联合验证；不自动扩seed，不按正式分数改候选。',
        '无论源判定，所有已注册行均完成原七视图，不选择性重跑。整体goal未完成：当前没有新联合行，也没有跨seed证明。',
        '', '实际文件：analysis_summary.json、seven_view_results.csv、formal_aggregate.csv、formal_strata.csv、paired_effects.csv、measured_costs.csv、complete_training_curves.csv、source_state_quality.csv、actual_G_coverage.csv。',
        '实际图：formal_results.png、complete_training_curves.png、source_strata_and_coverage.png、source_quality_and_existing_gate.png、measured_cost.png。图和表均由全量完成产物计算，不加载IQ、权重或target truth。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def harvest(output):
    spec=read(SPEC);root=spec['execution']['remote_run_root'];state=remote(state_script(root))
    if not ready(state):
        print(json.dumps(dict(status='FAILED' if state['failure'] else 'WAITING_LIVE' if state['live_pids'] else 'INCOMPLETE_NO_LIVE_HANDLE',
            live_pids=state['live_pids'],queue=state['queue'],failure=state['failure'])));return
    output=Path(output);output.mkdir(parents=True,exist_ok=False);write(output/'remote_completion.json',state)
    common=['complete.json','source_qualification.json','control_source_reuse.json','launch.json','source_smoke.json','source_panel.json']
    transfer((root,common),Path('.'),output)
    source=['completion.json','initialization.json','source_frozen.json','actual_native_args.json','resolved_config.json','epoch_metrics.jsonl',
        'metrics_epoch.csv','step_metrics.jsonl.gz','training_mechanism_progress.json','training_mechanism_complete.json','response_coverage_complete.json',
        'source_decision.json','source_panel_results.json','channel_execution.json']
    for row in spec['rows']:
        rid=row['row_id'];transfer((root+'/'+rid+'/source',source),Path(rid)/'source',output)
        for sub,names in [('prediction',['complete.json','resolved_config.json']),('scoring',['complete.json','scores.json'])]:
            transfer((root+'/'+rid+'/'+sub,names),Path(rid)/sub,output)
        transfer((root+'/'+rid,['complete.json']),Path(rid),output)
        transfer((root+'/logs',[rid+'.log']),Path('logs'),output)
    control=PurePosixPath(spec['rows'][0]['config']['baseline_reference']).parents[1]
    for arm in ('B00','R_LEGACY'):
        remote_row=str(control/(arm+'-s2026092701'))
        for sub,names in [('prediction',['complete.json']),('scoring',['complete.json','scores.json'])]:
            transfer((remote_row+'/'+sub,names),Path('controls')/arm/sub,output)
    result=analyze(output,spec,state)
    print(json.dumps(dict(status=result['status'],source_pass=result['source_qualification']['passed'],totals=result['totals'],formal_matrix=result['formal_matrix'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();harvest(a.output)
