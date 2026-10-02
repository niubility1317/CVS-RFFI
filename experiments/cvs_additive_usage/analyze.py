"""Independent full-row confusion/cell/log reconciliation; no target inputs."""
import argparse,csv,json,math,statistics
from pathlib import Path
from experiments.cvs_additive_usage.audit import RUN,SOURCE_RUN,MODES,VARIANTS,SEEDS,SOURCE_COMMIT,FULL_FP32_POLICY


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')
def csvwrite(p,rows):
    with Path(p).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def validate(d,source):
    completion=d['completion'];actual=d['resolved'];result=d['statistics']
    if completion['status']!='SOURCE_DIAGNOSTIC_COMPLETE' or completion['row_count']!=40 or completion['source_V_count_per_row']!=27000:raise ValueError('Incomplete diagnostic')
    for obj in (completion,result):
        if obj['source_commit']!=SOURCE_COMMIT or obj['target_access'] or obj['optimizer_steps'] or obj['model_updated'] or obj['used_for_selection'] or obj['training_augmentation'] or not obj['source_selection_unchanged']:raise ValueError('Source-only frozen diagnostic contract differs')
    if actual['backend_flags']!=FULL_FP32_POLICY or actual['L_s_use']!='not_iterated' or actual['U_s_use']!='unused' or actual['source_role']!='V':raise ValueError('Actual source/precision contract differs')
    if result['source_selection']!=source['source_selection']:raise ValueError('Current source choice changed')
    rows=result['rows'];expected={(v,s,m) for v in VARIANTS for s in SEEDS for m in MODES}
    if len(rows)!=40 or {(r['variant'],r['model_seed'],r['mode']) for r in rows}!=expected or len({r['row_id'] for r in rows})!=40:raise ValueError('Incomplete unique40row matrix')
    by={}
    for r in rows:
        cm=r['confusion'];n=sum(map(sum,cm));tp=[cm[i][i] for i in range(6)]
        if len(cm)!=6 or any(len(a)!=6 or any(type(v)!=int or v<0 for v in a) for a in cm) or n!=27000 or r['count']!=27000:raise ValueError('Source confusion/count malformed')
        f1=statistics.mean(2*tp[i]/(sum(cm[i])+sum(a[i] for a in cm)) if sum(cm[i])+sum(a[i] for a in cm)>0 else 0 for i in range(6))
        if r['accuracy']!=sum(tp)/n or abs(f1-r['macro_f1'])>1e-12 or any(not math.isfinite(r[k]) for k in ('ce','logit_response_max','accuracy','macro_f1')):raise ValueError('Source metrics differ from confusion')
        groups=r['groups'];keys={(g['tx'],g['receiver'],g['day']) for g in groups}
        if len(groups)!=90 or keys!={(t,rx,day) for t in range(6) for rx in (1,3,4,6,8) for day in (1,2,3)} or any(g['count']!=300 or not 0<=g['correct']<=300 or not 0<=g['changed_count']<=300 for g in groups):raise ValueError('Incomplete90 sourcecells')
        if sum(g['correct'] for g in groups)!=sum(tp) or sum(g['changed_count'] for g in groups)!=r['prediction_changed_count'] or not 0<=r['prediction_changed_count']<=n:raise ValueError('Cells differ from row totals')
        for t in range(6):
            if sum(cm[t])!=4500 or sum(g['correct'] for g in groups if g['tx']==t)!=tp[t]:raise ValueError('TX cell/confusion inconsistency')
        if r['mode']=='trained':
            original=next(a for a in source['rows'] if a['resolved']['variant']==r['variant'] and a['resolved']['model_seed']==r['model_seed'])
            if r['accuracy']!=original['completion']['final_source_metrics']['source_val_accuracy'] or r['prediction_changed_count']!=0 or r['logit_response_max']!=0:raise ValueError('Frozen trained original differs')
        by[(r['variant'],r['model_seed'],r['mode'])]=r
    lines=d['log_text'].splitlines();runtime=d['log_text'].split('RESOLVED_CONFIG ',1)
    if len(runtime)!=2 or any(s in runtime[1] for s in ('Traceback (most recent call last)','CUDA out of memory','FloatingPointError','Killed')):raise ValueError('Diagnostic runtime error/missing config')
    logged=[json.loads(s[4:]) for s in lines if s.startswith('ROW ')]
    if logged!=[{k:v for k,v in r.items() if k not in ('groups','confusion')} for r in rows] or not any(s.startswith('COMPLETE ') for s in lines):raise ValueError('Fulltext/row statistics differ')
    return by


def analyze(root):
    report=root/'automation_reports/CV-SincNet'/RUN;e=report/'evidence';d=read(e/'final_readback.json')
    source=read(root/'automation_reports/CV-SincNet'/SOURCE_RUN/'evidence/source_research_complete.json');by=validate(d,source)
    rows=[];cells=[];summaries=[];rx=[]
    for v in VARIANTS:
        for mode in MODES:
            values=[];changed=[]
            for seed in sorted(SEEDS):
                r=by[(v,seed,mode)];base=by[(v,seed,'trained')];delta=100*(r['accuracy']-base['accuracy']);values.append(delta);changed.append(r['prediction_changed_count'])
                rows.append({k:r[k] for k in ('row_id','variant','model_seed','mode','count','accuracy','macro_f1','ce','prediction_changed_count','logit_response_max')}|dict(accuracy_delta_pp=delta))
                cells.extend(dict(variant=v,model_seed=seed,mode=mode,**g) for g in r['groups'])
            summaries.append(dict(variant=v,mode=mode,accuracy_mean=statistics.mean(by[(v,s,mode)]['accuracy'] for s in SEEDS),accuracy_seed_sd=statistics.stdev(by[(v,s,mode)]['accuracy'] for s in SEEDS),
                delta_pp_mean=statistics.mean(values),delta_pp_seed_sd=statistics.stdev(values),delta_pp_by_seed=values,changed_mean=statistics.mean(changed)))
            for receiver in (1,3,4,6,8):
                values=[]
                for seed in sorted(SEEDS):
                    actual=[g for g in by[(v,seed,mode)]['groups'] if g['receiver']==receiver];base=[g for g in by[(v,seed,'trained')]['groups'] if g['receiver']==receiver]
                    values.append(100*(sum(g['correct'] for g in actual)-sum(g['correct'] for g in base))/5400)
                rx.append(dict(variant=v,mode=mode,receiver=receiver,delta_pp_mean=statistics.mean(values),delta_pp_seed_sd=statistics.stdev(values)))
    csvwrite(e/'all40_rows.csv',rows);csvwrite(e/'all3600_source_cells.csv',cells);csvwrite(e/'sameweight_mode_summary.csv',summaries);csvwrite(e/'all50_source_RX_effects.csv',rx)
    validation=dict(status='VERIFIED',rows=40,cells=3600,all_confusions_metrics_cells_recomputed=True,all_full_text_rows_reconciled=True,
                    all8_trained_rows_equal_original_E200=True,source_selection_unchanged=True,model_updated=False,target_access=False,used_for_selection=False,goal_achieved=False)
    write(e/'analysis_validation.json',validation);write(e/'mode_summary.json',summaries)
    text='# CVS 加性坐标实际使用：完整冻结源域消融报告\n\n8个合规E200模型×5种固定内部干预，40行每行27000个同物理源V样本、90单元；全部3600单元与40份混淆矩阵、Macro-F1、日志独立核对。8个trained行与原E200源准确率完全一致。没有优化、训练增强、模型状态更新、目标访问或源重排。\n\n'
    text+='## 同权重内部干预\n\n频率参考只置零两个圆周坐标；相干度参考只置零质量坐标（对应相干度1）。全部坐标参考保留学到的conditioner偏置；零加性项另移除输入响应与偏置方向注入。源波形及同步后的身份特征相同，仅修改固定权重的内部加性项输入。这不是物理TX/RX干预或唯一硬件参数估计。Δ为干预减trained，负值表示屏蔽该部分后准确率下降，不作选模。\n\n| 核心 | 干预 | 源V均值±seedSD（%） | Δ均值±SD（百分点） | 四seedΔ | 改变预测数均值/27000 |\n|---|---|---:|---:|---|---:|\n'
    for r in summaries:text+=f"| {r['variant']} | {r['mode']} | {100*r['accuracy_mean']:.4f}±{100*r['accuracy_seed_sd']:.4f} | {r['delta_pp_mean']:+.4f}±{r['delta_pp_seed_sd']:.4f} | "+', '.join(f'{x:+.4f}' for x in r['delta_pp_by_seed'])+f" | {r['changed_mean']:.2f} |\n"
    text+='\n完整逐seed CE/F1/变化数、每源RX效应及全部TX×RX×day单元均保存。内部依赖并不等于跨接收机因果身份信息，源V包含原5个源RX，不能称未知RX测试。该诊断不会改变本轮已经冻结的源选择，不宣称性能优化完成。\n\n'
    text+=f"实际源权重commit `{SOURCE_COMMIT}`；加载前核实完整源数据角色、scratch/E200及实际FP32，payload与source metadata逐字段相同且strict state加载。正式源IQ只在全部8份权重核实后读取；L/U没有迭代、V不拟合。每个模型全部干预后与加载时state逐张量完全一致，选择文件前后相同。一次矩阵owner按登记空闲GPU执行，耗时{d['completion']['elapsed_seconds']:.3f}s，峰值显存{d['completion']['peak_cuda_allocated_bytes']}bytes，硬件{d['resolved']['hardware']}；这是全部流程耗时，不是星载训练或单次推理benchmark。\n\n"
    text+='本次明确source-only诊断，不产生新clean预测/评分；不追加LEO、support、SFT、新类或D92三阶段。全部负结果保留，目标反馈禁令不变。\n\n[40行结果](evidence/all40_rows.csv) · [3600源单元](evidence/all3600_source_cells.csv) · [全部源RX效应](evidence/all50_source_RX_effects.csv) · [独立核对](evidence/analysis_validation.json) · [实际远端读回](evidence/final_readback.json)。\n'
    (report/'report.md').write_text(text,encoding='utf-8')
    write(e/'next_source_handoff.json',dict(run_id=RUN,status='SOURCE_ONLY_HANDOFF',mode_summary=summaries,validation=validation,
        target_scores_included=False,next_action='Use source-only mode effects and original source geometry to design next physics-grounded CE architecture;no target scores, same split, scratch, performancefirst selection/defaultselectedclean',goal_achieved=False))
    print(json.dumps(dict(validation=validation,summary=summaries),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
