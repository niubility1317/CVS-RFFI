"""Independent saved-source prediction/log analysis; no IQ/checkpoint/model imports."""
import argparse,csv,copy,json,math
from pathlib import Path
from experiments.cvs_f_crossrx_practical20.harvest import analyze as analyze_pair,collect,read,write,SCENES
ROOT=Path(__file__).resolve().parents[2]
RECEIVERS=(1,3,4,6,8)
FIELDS=('urban_gain_pp','low_gain_pp','urban_macroF1_gain_pp','clean_drop_pp','seen_drop_pp')

def confirmation(pairs,recorded):
    if set(pairs)!=set(RECEIVERS):raise ValueError('All five fixed held-source folds required')
    if not recorded['source_only'] or recorded['target_access'] or recorded['model_seed_count']!=1:raise ValueError('Source-only single-modelseed confirmation required')
    decisions={}
    for held,s in pairs.items():
        if s['target_access'] or s['model_seed_count']!=1 or s['source_fold_count']!=1 or set(s['models'])!={'C0','C1'}:raise ValueError('One source pair/one modelseed per held RX required')
        if any(v['held_RX']!=held or v['fit_RX']!=[r for r in RECEIVERS if r!=held] or v['target_access'] or v['U_truth_read'] or v['head_fit_updates'] or v['optimizer_updates']!=4440 for v in s['models'].values()):raise ValueError('Held-source role/result provenance differs')
        q=s['source_decision'];decisions[str(held)]=q
        if not q['source_only'] or q['target_access'] or q['status']!=('PASS' if all(q['checks'].values()) else 'FAIL'):raise ValueError('Per-fold source status and checks disagree')
        if q!=recorded['per_source_RX'][str(held)]:raise ValueError('Independently verified per-source decision differs from dispatcher')
    status='PASS' if all(v['status']=='PASS' for v in decisions.values()) else 'FAIL'
    means={k:sum(v[k] for v in decisions.values())/5 for k in FIELDS};worst=max(v['clean_drop_pp'] for v in decisions.values())
    if recorded['status']!=status or any(not math.isclose(means[k],recorded['five_fold_mean'][k],abs_tol=1e-9) for k in FIELDS) or not math.isclose(worst,recorded['worst_clean_drop_pp'],abs_tol=1e-9):raise ValueError('Independent fivefold status/mean/worst-drop differs')
    return dict(status=status,per_source_RX=decisions,five_fold_mean=means,worst_clean_drop_pp=worst,model_seed_count=1,source_fold_count=5,
        failed_source_RX=[r for r in RECEIVERS if decisions[str(r)]['status']!='PASS'],source_only=True,target_access=False,goal_completed=False,automatic_extension=False)

def summarize(output,pairs,state,reference_path):
    output=Path(output);q=confirmation(pairs,state['source_decision']);scenes=[];paired=[]
    for held in RECEIVERS:
        s=pairs[held]
        for row in s['scene_results']:scenes.append(dict(held_source_RX=held,**row))
        models=s['models'];b,a=(models[k] for k in ('C0','C1'));d=q['per_source_RX'][str(held)]
        paired.append(dict(held_source_RX=held,C0_clean=100*b['held_V']['accuracy'],C1_clean=100*a['held_V']['accuracy'],clean_gain_pp=-d['clean_drop_pp'],
            C0_low_urban=100*b['practical']['practical_low_urban']['accuracy'],C1_low_urban=100*a['practical']['practical_low_urban']['accuracy'],
            low_gain_pp=d['low_gain_pp'],urban_gain_pp=d['urban_gain_pp'],urban_F1_gain_pp=d['urban_macroF1_gain_pp'],seen_drop_pp=d['seen_drop_pp'],source_status=d['status']))
    scene_means=[]
    for scene in SCENES:
        per={arm:[r for r in scenes if r['scene']==scene and r['arm']==arm] for arm in ('C0','C1')}
        if any(len(rr)!=5 for rr in per.values()):raise ValueError('Exactly five source-fold rows per arm/scene required')
        row=dict(scene=scene)
        for arm,rr in per.items():
            row[arm+'_mean_accuracy']=sum(v['accuracy'] for v in rr)/5;row[arm+'_mean_macroF1']=sum(v['macro_f1'] for v in rr)/5
            worst=min(rr,key=lambda v:v['accuracy']);row[arm+'_worst_RX']=worst['held_source_RX'];row[arm+'_worst_accuracy']=worst['accuracy']
        row['accuracy_gain_pp']=100*(row['C1_mean_accuracy']-row['C0_mean_accuracy']);row['macroF1_gain_pp']=100*(row['C1_mean_macroF1']-row['C0_mean_macroF1']);scene_means.append(row)
    new_cost=sum(pairs[r]['real_forward_packets'] for r in RECEIVERS if r!=1)
    result=dict(status='ANALYZED',source_decision=q,paired_source_RX=paired,scene_means=scene_means,
        new_model_epochs=160,new_updates=35520,new_real_forward_packets=new_cost,reused_RX1_model_epochs=40,reused_RX1_updates=8880,reused_RX1_real_forward_packets=pairs[1]['real_forward_packets'],
        new_matrix_wall_seconds=state['complete']['wall_seconds'],reference_summary=str(reference_path),actual_training_commit=state['submit']['commit'],
        model_seed_count=1,source_fold_count=5,formal_test_status='NOT_REGISTERED_SOURCE_ONLY',target_access=False,goal_completed=False,automatic_extension=False)
    for name,rows in [('paired_source_RX.csv',paired),('all_source_scene_results.csv',scenes),('source_scene_means.csv',scene_means)]:
        with (output/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)));writer.writeheader();writer.writerows(rows)
    plot(output,paired);write(output/'analysis_summary.json',result)
    lines=['# F＋跨 RX 配对星地：五源折确认结果','',f"源确认 **{q['status']}**。八个新 scratch E20 加已冻结 RX1 参考，一个模型 seed、五个源 RX 留出折；0 target。",'',
        '|held RX|C0 clean|C1 clean|clean Δ/pp|C0 low|C1 low|low Δ/pp|urban Δ/pp|源规则|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for v in paired:lines.append(f"|{v['held_source_RX']}|{v['C0_clean']:.4f}%|{v['C1_clean']:.4f}%|{v['clean_gain_pp']:+.4f}|{v['C0_low_urban']:.4f}%|{v['C1_low_urban']:.4f}%|{v['low_gain_pp']:+.4f}|{v['urban_gain_pp']:+.4f}|{v['source_status']}|")
    lines+=['',f"固定失败源折：{q['failed_source_RX']}。五折平均 clean Δ={-q['five_fold_mean']['clean_drop_pp']:+.6f} pp、low Δ={q['five_fold_mean']['low_gain_pp']:+.6f} pp、urban Δ={q['five_fold_mean']['urban_gain_pp']:+.6f} pp；最差 clean 损失={q['worst_clean_drop_pp']:.6f} pp。所有折必须满足原规则，平均阳性不覆盖失败折。",'',
        '|场景|C0 源RX均值|C1 源RX均值|Δ/pp|C0 最差RX/准确率|C1 最差RX/准确率|','|---|---:|---:|---:|---|---|']
    for v in scene_means:lines.append(f"|{v['scene']}|{100*v['C0_mean_accuracy']:.4f}%|{100*v['C1_mean_accuracy']:.4f}%|{v['accuracy_gain_pp']:+.4f}|RX{v['C0_worst_RX']}/{100*v['C0_worst_accuracy']:.4f}%|RX{v['C1_worst_RX']}/{100*v['C1_worst_accuracy']:.4f}%|")
    lines+=['',f"新增160模型轮／35520更新／{new_cost}网络包，复用RX1参考40模型轮／8880更新／{pairs[1]['real_forward_packets']}网络包单列。父权重成本0，各模型从零E20；本次八行确认矩阵墙钟{result['new_matrix_wall_seconds']:.6f}s，不是每对独立墙钟，不能把四个分折报告的该数值相加。",'',
        '所有新行各完整20轮／4440步／native CSV／stdout／SupCon与F物理clip、同折全5994条L/U顺序、同流源physics和默认头预测独立复算。逐折训练曲线、状态/质量/TX/day分层、修复/新增错误和固定C0 clean正确集合保持率保存在RX3/RX4/RX6/RX8目录。RX1使用已完整独立核实的冻结参考，不重新训练或读其checkpoint。', '',
        '均值是五个源RX等权平均，F1也是逐源RX宏F1均值，最差RX单列。五折共享同一model seed和同一原源数据池，训练集重叠；不能给出跨modelseed复现或把五折当成五个独立seed。全source27000个V包各在其留出模型上评估，绝不把seen V当held V。', '',
        '未修改CVS、默认头或正式practical IQ/信道/SNR。没有新Ag、额外对齐/课程/U增强。此处不能证明独立Ag有效，也不等于正式clean+6practical或lowurban>60%、suburban距自身clean≤2pp的联合目标。预算耗尽不等于收敛。任一失败折保留，不换RX/seed或调整配方救援，不用目标反馈选择。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result

def analyze(output,spec,state,reference_summary,cached=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if state['dispatcher_live'] or state['failure'] or state['row_failures'] or not state['complete'] or state['complete']['status']!='ARTIFACTS_COMPLETE' or any(v['live'] for v in state['rows']):raise ValueError('All eight original workers must be complete before global confirmation')
    pairs={1:read(reference_summary)}
    for held in RECEIVERS[1:]:
        pair_spec=copy.deepcopy(spec);pair_spec['rows']=[r for r in spec['rows'] if r['config']['held_RX']==held]
        if len(pair_spec['rows'])!=2:raise ValueError('Exactly one C0/C1 pair per held source required')
        where=output/f'RX{held}'
        if not cached:collect(where,pair_spec)
        pair_state=dict(state,source_decision=state['source_decision']['per_source_RX'][str(held)])
        pairs[held]=analyze_pair(where,pair_spec,pair_state,held_RX=held)
        pairs[held]['wall_time_scope']='Entire eight-row confirmation matrix, not this pair alone';write(where/'analysis_summary.json',pairs[held])
        text=(where/'report.md').read_text(encoding='utf-8').replace('矩阵墙钟','全八行确认矩阵墙钟（非本折单独计时）')
        (where/'report.md').write_text(text,encoding='utf-8')
    return summarize(output,pairs,state,reference_summary)

def plot(output,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax,key,label,threshold in zip(axes,('clean_gain_pp','low_gain_pp','urban_gain_pp'),('Clean gain (pp)','Low urban gain (pp)','Urban mean gain (pp)'),(-.5,5.,3.)):
        colors=['#4477aa' if v[key]>=0 else '#cc6677' for v in rows];ax.bar([str(v['held_source_RX']) for v in rows],[v[key] for v in rows],color=colors)
        ax.axhline(0.,color='black',lw=.7);ax.axhline(threshold,color='grey',ls='--',lw=.9);ax.set(xlabel='Held original source RX (one model seed)',ylabel=label)
    fig.savefig(Path(output)/'source_fold_confirmation.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--state',required=True);p.add_argument('--reference-summary',required=True);p.add_argument('--cached',action='store_true');a=p.parse_args()
    spec=read(ROOT/'experiments/cvs_f_crossrx_folds20/registration.json');s=analyze(a.output,spec,read(a.state),a.reference_summary,a.cached)
    print(json.dumps({k:v for k,v in s.items() if k not in ('paired_source_RX','scene_means')},ensure_ascii=False))
