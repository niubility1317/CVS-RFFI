"""Complete paired source factorial comparisons without launching confirmation."""
import argparse
import json
from pathlib import Path
import statistics as st
from report_core90_v2_completed import table,write_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder)
    data=json.loads((folder/'summary.json').read_text());rows=data['run_summary'];assert len(rows)==21
    by={(r['row'],r['seed']):r for r in rows}
    comparisons=[]
    for left,right in [('A','B'),('A','C'),('B','D'),('A','E'),('B','F'),('C','D'),('E','F')]:
        values=[100*(by[('V2_'+right,s)]['mean_leo_accuracy']-by[('V2_'+left,s)]['mean_leo_accuracy']) for s in (392005,392006,392007)]
        comparisons.append(dict(comparison=right+'_minus_'+left,seed392005_pp=values[0],seed392006_pp=values[1],seed392007_pp=values[2],mean_pp=st.mean(values),sample_std_pp=st.stdev(values)))
    write_csv(folder/'factorial_deltas.csv',comparisons)
    candidates=[r for r in rows if 'STRONG_SOURCE' in r['row']]
    winner=max(candidates,key=lambda r:(r['mean_four_macro_f1'],-{'LOW':1e-4,'BASE':2e-4,'HIGH':4e-4}[r['row'].split('_')[-1]]))
    result=dict(comparisons=comparisons,source_candidate_winner=winner['run_id'],criterion='mean_four_macro_f1',
        value=winner['mean_four_macro_f1'],status='SOURCE_SELECTED_NOT_MULTI_SEED_CONFIRMED',target_used=False)
    (folder/'factorial_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    report=folder/'report.md';text=report.read_text(encoding='utf-8')
    text+='\n## 11. 完整solver×adv配对对比\n\n单位为LEO平均Accuracy百分点，正值表示后者更高。\n\n'
    text+=table(['比较','392005','392006','392007','平均Δ','SD'],[[r['comparison'],*[f"{r[k]:+.3f}" for k in ['seed392005_pp','seed392006_pp','seed392007_pp','mean_pp']],f"{r['sample_std_pp']:.3f}"] for r in comparisons])
    text+='\nE（EG、adv=0）在固定lr=2e-4的六个方法组中source LEO均值最高。D相对B的三个seed均下降；C与A三个seed预测一致，不能把B8执行计数当成收益。F相对E的三个seed均下降，说明对抗项收益依赖求解器。以上为source描述性差分，不主张目标域泛化或统计显著性。\n'
    text+='\n## 12. source学习率候选\n\n'
    text+=table(['候选','seed','LEO Accuracy','四场景Macro-F1'],[[r['row'],r['seed'],f"{r['mean_leo_accuracy']*100:.3f}",f"{r['mean_four_macro_f1']*100:.3f}"] for r in candidates])
    text+=f"\n按预登记四场景Macro-F1平均值选择，候选为{winner['run_id']}，lr=4e-4；该候选只有一个seed，尚未进行独立多seed确认。不得把其最优单seed分数与E组三seed均值直接当成公平优劣结论。本次未追加训练、未进行target评估。\n\n[配对差分CSV](factorial_deltas.csv) · [完整比较摘要](factorial_summary.json)\n"
    report.write_text(text,encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
