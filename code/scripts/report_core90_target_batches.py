"""Combine completed frozen test batches without rerunning or selecting models."""
import argparse,csv,json,math
from pathlib import Path

NEW=('B3_head_scale','B3_adv_low','B3_adv_high','B4','B4_fixedk','B5','B8','S2','S2_random','C3')

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();scores={};batches=[];reference=None
    for folder in a.inputs:
        root=Path(folder)
        m=json.loads((root/'frozen_manifest.json').read_text(encoding='utf-8'))
        c=json.loads((root/'complete.json').read_text(encoding='utf-8'))
        assert c['complete'] and c['rows']==m['rows']
        if reference:
            for key in ('target_rxs','target_days','samples_per_scene','scenes','batch_size','augmentation_seed','num_classes','seed'):
                assert m[key]==reference[key],key
        reference=m;batches.append(dict(manifest=m,completion=c))
        for row in m['rows']:
            assert row not in scores,'Duplicate row across batches'
            s=json.loads((root/(row+'_target_scores.json')).read_text(encoding='utf-8'))
            assert s['complete'] and s['target_evaluated'] and not s['source_only']
            assert s['samples_per_scene']==m['samples_per_scene'] and s['prediction_count']==m['samples_per_scene']*4
            for v in s['scenes'].values():
                assert sum(x['count'] for x in v['per_rx'].values())==v['count']==m['samples_per_scene']
                assert sum(x['count'] for x in v['per_day'].values())==v['count']
                assert sum(v['per_tx_count'].values())==v['count'] and not v['missing_registered_classes']
                assert math.isclose(sum(x['accuracy']*x['count'] for x in v['per_rx'].values())/v['count'],v['accuracy'],abs_tol=1e-12)
            scores[row]=s
    assert set(NEW)<=set(scores)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    stem='core90_target25_392005_20260911';records=[]
    for row,s in scores.items():
        for scene,v in s['scenes'].items():
            for scope,groups in [('overall',{'all':v}),('receiver',v['per_rx']),('day',v['per_day'])]:
                for group,g in groups.items():
                    records.append(dict(row=row,new_since_first15=row in NEW,scene=scene,scope=scope,group=group,tx='',count=g['count'],accuracy_pct=g['accuracy']*100,macro_f1_pct=g['macro_f1']*100,macro_recall_pct=g['macro_recall']*100))
                    for tx,acc in g['per_tx_accuracy'].items():
                        records.append(dict(row=row,new_since_first15=row in NEW,scene=scene,scope=scope+'_tx',group=group,tx=tx,count=g['per_tx_count'][tx],accuracy_pct=acc*100,macro_f1_pct='',macro_recall_pct=''))
    with (out/(stem+'.csv')).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/(stem+'.json')).write_text(json.dumps(dict(batches=batches,scores=scores,new_rows=NEW),ensure_ascii=False,indent=2),encoding='utf-8')
    scenes=reference['scenes']
    lines=['# CORE90新增10行与累计25行目标测试数据','',
        '本次新增训练完成10行；复用B8已完成测试，对另外9行新做测试。先前15行未重跑。各批内部均先完成所有模型四场景预测再评分，三批完成时间不同；不能声称25行在同一时刻统一封存。全部模型与测试设置冻结，无测试反馈调参。',
        '',f"测试集每场景{reference['samples_per_scene']:,}条；RX={reference['target_rxs']}，day={reference['target_days']}，6个注册TX。每RX24000、每day42000、每TX28000条。训练seed392005、增强seed{reference['augmentation_seed']}、batch256。百分数保留两位，CSV/JSON保留完整精度。"]
    for title,which in [('新增10行',NEW),('累计25行',list(scores))]:
        lines+=['','## '+title,'','|模型|clean|晴空LEO|低仰角LEO|雨衰LEO|LEO均值|最弱LEO接收机|','|---|---:|---:|---:|---:|---:|---:|']
        for row in which:
            s=scores[row];vals=[s['scenes'][c]['accuracy'] for c in scenes]+[s['leo_mean_accuracy'],min(s['scenes'][c]['worst_rx_accuracy'] for c in scenes[1:])]
            lines.append('|'+row+'|'+'|'.join(f'{v*100:.2f}' for v in vals)+'|')
    for scene in scenes:
        lines+=['',f'## 新增10行：{scene}详细指标','','|模型|Macro-F1|Macro-recall|最弱RX|最弱RX准确率|','|---|---:|---:|---|---:|']
        for row in NEW:
            v=scores[row]['scenes'][scene];rx=min(v['per_rx'],key=lambda x:v['per_rx'][x]['accuracy'])
            lines.append(f"|{row}|{v['macro_f1']*100:.2f}|{v['macro_recall']*100:.2f}|RX{rx}|{v['per_rx'][rx]['accuracy']*100:.2f}|")
        for field,groups,label in [('per_rx',reference['target_rxs'],'RX'),('per_day',reference['target_days'],'day'),('per_tx_accuracy',range(6),'TX')]:
            lines+=['',f'### {scene}逐{label}准确率','','|模型|'+'|'.join(label+str(g) for g in groups)+'|','|---|'+'---:|'*len(groups)]
            for row in NEW:
                v=scores[row]['scenes'][scene][field]
                vals=[v[str(g)] if field=='per_tx_accuracy' else v[str(g)]['accuracy'] for g in groups]
                lines.append('|'+row+'|'+'|'.join(f'{x*100:.2f}' for x in vals)+'|')
    lines+=['','## 验证与边界','',f'累计{len(scores)}模型、{sum(s["prediction_count"] for s in scores.values()):,}条预测；CSV共{len(records)}行。各row四场景的RX加权准确率、样本计数、TX/day计数和注册类覆盖均核对通过。',
        '', '单seed闭集描述性结果，不支持多seed统计显著性、未知类性能或科学晋级结论。B6尚未纳入测试，不能将训练中的数据混入完整结果。']
    (out/(stem+'.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(models=len(scores),csv_rows=len(records),new_rows=NEW)))

if __name__=='__main__':main()
