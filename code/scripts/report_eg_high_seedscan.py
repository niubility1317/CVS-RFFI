"""Render a bounded source-only EG/high-LR scan, including unfinished rows."""
import argparse,json,statistics as st,subprocess,sys
from pathlib import Path
from report_core90_v2_completed import table
from collect_core90_v2_completed import SCENES

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);a=p.parse_args()
    path=Path(a.audit);d=json.loads(path.read_text(encoding='utf-8'));out=path.parent
    subprocess.run([sys.executable,'-X','utf8',str(Path(__file__).with_name('report_core90_v2_completed.py')),'--audit',str(path),'--tables-only'],check=True,capture_output=True,text=True,encoding='utf-8')
    rows=d['rows'];done=[r for r in rows if r['queue_state']=='completed']
    lines=['# 完整EG＋高学习率seed扫描：当前完整可用结果','',f"截止{d['snapshot_finished']}，{len(done)}/{len(rows)}完成；本报告全部为source V最终评分，目标测试尚未执行。",'',
    '每场景27000样本，四场景108000决策/模型，source RX1/3/4/6/8，day1/2/3。全部模型lr=0.0004、scratch E200，final_only。目标基准已有研究者接触，本轮为探索性后续训练。','',
    table(['模型','状态','最后完整epoch','最新动作epoch'],[[r['run_id'],r['queue_state'],r['epochs'][-1]['epoch'],r['actions']['last_epoch']] for r in rows])]
    for metric in ('accuracy','macro_f1'):
        lines += ['',f'## {metric}（%）','',table(['模型',*SCENES,'LEO均值'],[[r['run_id'],*[f"{100*r['independent_scores']['scenes'][s][metric]:.3f}" for s in SCENES],f"{100*st.mean(r['independent_scores']['scenes'][s][metric] for s in SCENES[1:]):.3f}"] for r in done])]
    lines+=['','## 分组统计','', '均值±样本标准差；adv0.35若未完成三个seed，不视为完整三seed结果。','']
    agg=[]
    for adv in (0.,.35):
        group=[r for r in done if r['config']['lambda_adv']==adv]
        for metric in ('accuracy','macro_f1'):
            values=[100*st.mean(r['independent_scores']['scenes'][s][metric] for s in SCENES[1:]) for r in group]
            if values:agg.append([adv,len(values),metric,f'{st.mean(values):.3f}',f'{st.stdev(values):.3f}' if len(values)>1 else 'N/A'])
    lines += [table(['adv','已完成seed数','LEO指标','均值','SD'],agg),'','## 验证与文件','',
    f"全量解析{sum(len(r['epochs']) for r in rows)}个epoch、{sum(r['actions']['records'] for r in rows)}条动作；已完成{len(done)*108000}预测独立重算，详见full_source_audit.json。",'',
    'scene_metrics.csv包含总体Precision/Recall/F1与floor；per_rx.csv、per_day.csv、per_tx.csv及confusion_matrices.csv保存全部分组结果。epoch_metrics_all_rows.csv保存全部现有epoch，matrix_progress.csv保存本次进度。','',
    '训练完成不等于目标测试完成。最后一行健康训练继续，本次没有启动、停止或重跑实验。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[:14]));print(agg)
    for r in rows:
        if r['queue_state']=='active':
            times=[e['epoch_seconds'] for e in r['epochs'][-10:]]
            print('ETA_remaining_training_minutes',st.mean(times)*(200-r['epochs'][-1]['epoch'])/60)

if __name__=='__main__':main()
