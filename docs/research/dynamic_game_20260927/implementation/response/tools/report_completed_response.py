"""Complete tabular exports and independent recount of fixed test predictions."""
import argparse,csv,json,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.joint_metrics import classification
SCENES=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']

def export(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    state=json.loads((a.artifacts/'state.json').read_text());assert state['status']=='SCORED_COMPLETE'
    a.output.mkdir(parents=True,exist_ok=False)
    truth=json.loads((a.artifacts/'truth_sidecar.json').read_text());labels={r['sample_id']:r['label'] for r in truth['records']}
    ids=sorted(labels);y=np.array([labels[k] for k in ids]);vectors={};overview=[];groups=[];curves=[];verified=[]
    for rid in state['rows']:
        folder=a.artifacts/rid;score=json.loads((folder/'score.json').read_text());records=json.loads((folder/'prediction/predictions.json').read_text())['records']
        assert all(r['run_id']==state['run_id'] and r['row_id']==rid for r in records)
        vec={}
        for scene in SCENES:
            pairs=[(r['sample_id'],r['predicted_class']) for r in records if r['scenario']==scene]
            values=dict(pairs);assert len(pairs)==len(values)==168000 and set(values)==set(labels)
            pred=np.array([values[k] for k in ids]);vec[scene]=pred
            rec=classification(y,pred,6);expected=score['groups']['scene'][scene]
            assert rec['confusion']==expected['confusion'] and rec['accuracy']==expected['accuracy']
            verified.append(dict(row=rid,scene=scene,n=168000,independent_confusion_recount='PASS'))
        vectors[rid]=vec
        logs=[json.loads(x) for x in (folder/'logs.jsonl').read_text().splitlines() if x.strip()]
        assert [x['epoch'] for x in logs]==list(range(1,201))
        assert all(x['accepted']==222 and x['total_step']==222*x['epoch'] for x in logs)
        for row in logs:curves.append(dict(row_id=rid,epoch=row['epoch'],accepted=row['accepted'],total_step=row['total_step'],mean_loss=row['mean_loss'],source_V_accuracy=row['source_validation']['accuracy'],source_V_CE=row['source_validation']['ce']))
        completion=json.loads((folder/'completion.json').read_text());cfg=json.loads((folder/'resolved_config.json').read_text());j=cfg['joint']
        row=dict(row=rid,method=rid.rsplit('_s',1)[0],seed=j['model_seed'],epochs=200,steps=44400,lr=cfg['lr'],weight_decay=cfg['weight_decay'],training_hours=completion['elapsed_seconds']/3600,source_V_accuracy=logs[-1]['source_validation']['accuracy'])
        for scene in SCENES:
            row[scene+'_accuracy']=score['groups']['scene'][scene]['accuracy'];row[scene+'_macro_f1']=score['groups']['scene'][scene]['macro_f1']
        row.update(overall_four_scene_accuracy=score['overall']['accuracy'],overall_four_scene_macro_f1=score['overall']['macro_f1'],leo_mean_accuracy=float(np.mean([row[s+'_accuracy'] for s in SCENES[1:]])),leo_mean_macro_f1=float(np.mean([row[s+'_macro_f1'] for s in SCENES[1:]])),worst_RX_leo_mean=score['worst_RX_leo_mean'],worst_TX_leo_mean=score['worst_TX_leo_mean'],worst_RX_scene_accuracy=score['worst_RX_scene_accuracy'],group_risk_cvar=score['group_risk_cvar'])
        overview.append(row)
        for axis,cells in score['groups'].items():
            for key,val in cells.items():groups.append(dict(row=rid,axis=axis,key=key,n=val['n'],accuracy=val['accuracy'],macro_f1=val['macro_f1'],confusion=json.dumps(val['confusion']),per_class_f1=json.dumps(val['per_class_f1'])))
        # Add missing day/scene and all two-way RX/day/TX aggregates from full joint cells.
        cube=score['groups']['RX_day_TX_scene']
        for name,keep in [('day_scene',(1,3)),('RX_day_scene',(0,1,3)),('RX_TX_scene',(0,2,3)),('day_TX_scene',(1,2,3))]:
            sums={}
            for key,val in cube.items():
                parts=key.split('|');k='|'.join(parts[i] for i in keep);sums[k]=sums.get(k,np.zeros((6,6),dtype=int))+np.asarray(val['confusion'])
            for key,cm in sums.items():
                den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(6),where=den!=0)
                groups.append(dict(row=rid,axis=name,key=key,n=int(cm.sum()),accuracy=float(np.trace(cm)/cm.sum()),macro_f1=float(f1.mean()),confusion=json.dumps(cm.tolist()),per_class_f1=json.dumps(f1.tolist())))
    paired=[]
    contrasts=[(f'{c}_s{seed}',f'{r}_s{seed}') for seed in (392005,392006,392007)
        for c,r in [('DRIC','SIM'),('EG','SIM'),('TR_EG','EG'),('XT_DANN','EG'),('CF_EG','EG'),
            ('R0-10','R0-00'),('R0-11','R0-01'),('R0-01','R0-00'),('R0-11','R0-10'),('R0-10','ADV0-0'),('R0-11','ADV0-1')]]
    for candidate,reference in contrasts:
        if candidate not in vectors or reference not in vectors:continue
        for scene in SCENES:
            c=vectors[candidate][scene];r=vectors[reference][scene];rescue=int(((c==y)&(r!=y)).sum());harm=int(((c!=y)&(r==y)).sum())
            paired.append(dict(candidate=candidate,reference=reference,scene=scene,n=168000,rescue=rescue,harm=harm,net=rescue-harm,accuracy_difference_pp=100*(rescue-harm)/168000))
        leo=[x for x in paired if x['candidate']==candidate and x['reference']==reference and x['scene'] in SCENES[1:]]
        rescue=sum(x['rescue'] for x in leo);harm=sum(x['harm'] for x in leo)
        paired.append(dict(candidate=candidate,reference=reference,scene='LEO_MEAN',n=504000,rescue=rescue,harm=harm,net=rescue-harm,accuracy_difference_pp=100*(rescue-harm)/504000))
    paired_summary=[]
    keys=sorted({(x['candidate'].rsplit('_s',1)[0],x['reference'].rsplit('_s',1)[0],x['scene']) for x in paired})
    for candidate,reference,scene in keys:
        selected=[x for x in paired if x['candidate'].rsplit('_s',1)[0]==candidate and x['reference'].rsplit('_s',1)[0]==reference and x['scene']==scene]
        values=[x['accuracy_difference_pp'] for x in selected]
        paired_summary.append(dict(candidate=candidate,reference=reference,scene=scene,n_seeds=len(values),mean_difference_pp=float(np.mean(values)),sd_difference_pp=float(np.std(values,ddof=1)) if len(values)>1 else 'NA'))
    export(a.output/'paired_seed_summary.csv',paired_summary)
    aggregate=[]
    for method in sorted({x['method'] for x in overview}):
        selected=[x for x in overview if x['method']==method]
        for metric in [s+'_'+m for s in SCENES for m in ('accuracy','macro_f1')]+['leo_mean_accuracy','leo_mean_macro_f1']:
            values=[x[metric] for x in selected];aggregate.append(dict(method=method,metric=metric,n_seeds=len(values),seeds='|'.join(str(x['seed']) for x in selected),mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else 'NA'))
    lookup={r['row']:r for r in overview};interaction=[]
    for seed in (392005,392006,392007):
        names={cell:f'R0-{cell}_s{seed}' for cell in ('00','01','10','11')}
        if not all(x in lookup for x in names.values()):continue
        for metric in [s+'_'+m for s in SCENES for m in ('accuracy','macro_f1')]+['leo_mean_accuracy','leo_mean_macro_f1']:
            v={c:lookup[n][metric] for c,n in names.items()}
            interaction.append(dict(seed=seed,metric=metric,DR_effect_SIM_pp=100*(v['10']-v['00']),DR_effect_EG_pp=100*(v['11']-v['01']),
                EG_effect_DRoff_pp=100*(v['01']-v['00']),EG_effect_DRon_pp=100*(v['11']-v['10']),interaction_pp=100*(v['11']-v['10']-v['01']+v['00'])))
    if interaction:export(a.output/'DR_EG_interaction.csv',interaction)
    inventory=json.loads(a.inventory.read_text(encoding='utf-8'))
    assert state['run_id']==inventory['state']['run_id']
    assert len(inventory['state']['rows'])==36
    assert set(state['rows'])=={k for k,v in inventory['state']['rows'].items() if v['status']=='TRAINING_COMPLETE'}
    coverage=[dict(row=k,training_status=v['status'],test_status='SCORED' if k in vectors else 'NOT_EVALUATED_INCOMPLETE',steps=inventory['workers'].get(k,{}).get('accepted_records',0)) for k,v in inventory['state']['rows'].items()]
    artifacts=[dict(row=rid,checkpoint=str(a.artifacts/rid/'final_ssdg.pth'),predictions=str(a.artifacts/rid/'prediction/predictions.json'),score=str(a.artifacts/rid/'score.json'),source_log=str(a.artifacts/rid/'logs.jsonl')) for rid in state['rows']]
    export(a.output/'artifact_paths.csv',artifacts)
    for name,rows in [('summary',overview),('all_groups',groups),('paired_rescue_harm',paired),('seed_summary',aggregate),('source_curves',curves),('coverage36',coverage),('independent_recount',verified)]:export(a.output/(name+'.csv'),rows)
    (a.output/'summary.json').write_text(json.dumps(dict(rows=overview,paired=paired,seeds=aggregate,interaction=interaction,verification=verified,coverage=coverage),indent=2)+'\n',encoding='utf-8')
    text=['# 已完成实验测试集完整数据','',f'本次汇总{len(overview)}个E200 checkpoint，共{len(overview)*672000:,}条预测；每模型4场景×168,000物理样本。{len(overview)*4}个场景的匿名ID覆盖和混淆矩阵独立重计全部通过。复用既有{len(state.get("reused_rows",[]))}行封存结果，其余为本次新增评估。','',
        '完成范围冻结于'+datetime.fromtimestamp(inventory['time'],timezone(timedelta(hours=8))).isoformat()+'；仅使用该快照中完成的固定E200，不选择中途或最佳测试checkpoint。','',
        '实验代号：R0-00/01分别为原生DR关闭下的SIM/完整EG，R0-10/11分别为原生DR开启下的SIM/完整EG。ADV0-0/1保留原生DR，但分别在SIM/EG下移除整项对抗损失。新路线SIM/EG采用reuse_origin_used_scales，旧R0/ADV0采用legacy_reestimate；跨这两套基线的差异不能直接归因于同一个单独机制。','',
        '这是既有测试集的描述性重测；不回流训练、选模或重跑，不作为新盲测确认。其余实验仍在训练或排队，见coverage36.csv。使用本地RTX5070Ti、Torch'+state['torch']+'，batch256、sat_seed392005；所有模型同评估配置，训练仍在N607。','',
        '共同训练预算：scratch、E200×222接受步、AdamW、LR0.0002、WD0.0001、L128/U256、clip5。耗时为原训练实际墙钟，不含本次推理。固定E200结果，不替换为最佳测试轮次。测试范围为RX0/2/5/7/9/10/11、day0/1/2/3、TX0–5；同168000物理样本分别给出clean与三LEO压力视图，不能将其当成672000个独立物理样本。','',
        '|实验|seed|Clean准确率|Clear|Low-elev|Rain|LEO均值|LEO Macro-F1均值|训练小时|','|--|--:|--:|--:|--:|--:|--:|--:|--:|']
    for r in overview:text.append('|'+r['method']+'|'+str(r['seed'])+'|'+ '|'.join(f'{100*r[k]:.4f}%' for k in [s+'_accuracy' for s in SCENES]+['leo_mean_accuracy','leo_mean_macro_f1'])+f"|{r['training_hours']:.2f}|")
    lookup={r['row']:r for r in overview}
    text+=['','同seed392005下，DRIC−SIM的LEO均值变化为'+f"{100*(lookup['DRIC_s392005']['leo_mean_accuracy']-lookup['SIM_s392005']['leo_mean_accuracy']):+.4f}"+'个百分点，Clean变化为'+f"{100*(lookup['DRIC_s392005']['clean_accuracy']-lookup['SIM_s392005']['clean_accuracy']):+.4f}"+'个百分点；训练墙钟分别为'+f"{lookup['DRIC_s392005']['training_hours']:.2f}/{lookup['SIM_s392005']['training_hours']:.2f}"+'小时。该墙钟包含并发资源影响，不能当作纯算法FLOPs对比。',
        'R0-10−R0-00的LEO均值变化为'+f"{100*(lookup['R0-10_s392005']['leo_mean_accuracy']-lookup['R0-00_s392005']['leo_mean_accuracy']):+.4f}"+'个百分点，Clean变化为'+f"{100*(lookup['R0-10_s392005']['clean_accuracy']-lookup['R0-00_s392005']['clean_accuracy']):+.4f}"+'个百分点。它支持该单seed、simultaneous条件下原生DR增量的观察，不能代替DR×完整EG协同结论。','',
        f'数据文件：summary.csv为每行完整指标与弱RX/TX；all_groups.csv为RX/day/TX及交叉场景分组、混淆矩阵和各类F1；source_curves.csv包含全部{len(overview)}×200轮源域曲线；seed_summary.csv包含已完成seed的均值与样本SD；paired_rescue_harm.csv为同seed配对；DR_EG_interaction.csv为已齐全四格的交互效应；coverage36.csv保留完整矩阵状态。CSV准确率/F1均用0–1小数，差值列为百分点。','',
        '单TX或TX交叉切片的Macro-F1按固定6类口径计算；该切片只含一个真实类，不能与全类别Macro-F1直接比较。优先查看其accuracy/召回与混淆矩阵。LEO均值是三场景算术平均，不将三场景当作独立训练seed。','',
        '已完成seed数：'+'；'.join(f"{method}={sum(x['method']==method for x in overview)}" for method in sorted({x['method'] for x in overview}))+ '。不完整方法不填0，单seed不计算SD。四格交互只在同seed四行均完成时计算；不能据单seed或两seed宣称三seed稳定优胜。DRIC−SIM、TR/XT/CF−EG及原生四格分别在对应控制范围内解释；旧ADV0连同域头损失移除，不能冒充仅关闭编码器对抗的对照。','',
        '原始checkpoint、预测、逐行score和日志：'+str(a.artifacts)+'。全部预测固定后才下载truth并由独立scorer连接；完整ID×场景覆盖由scorer校验，另按原始预测独立重计全部场景混淆矩阵。']
    (a.output/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(overview),predictions=len(overview)*672000,group_rows=len(groups),source_epochs=len(curves),recount='PASS')))

if __name__=='__main__':main()
