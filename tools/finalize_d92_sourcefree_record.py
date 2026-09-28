"""Archive the complete SFHead outcome without changing or rerunning the candidate."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
RUN='20260928-phase2-d92-sfhead-confirmation-manytx-m4-r01'

def main():
    rel=Path('automation_reports/CV-SincNet')/RUN;folder=WS/rel
    result=folder/'results'
    summary=json.loads((result/'summary/summary.json').read_text(encoding='utf-8'))
    arithmetic=json.loads((result/'arithmetic_audit.json').read_text(encoding='utf-8'))
    optimizer=json.loads((result/'optimizer_audit.json').read_text(encoding='utf-8'))
    if summary['rows']!=2412 or arithmetic['status']!='VERIFIED' or optimizer['total_fits']!=1200:
        raise ValueError('Incomplete independent analysis')
    # This script describes the fixed failed version; never use it to declare a different version successful.
    if summary['preregistered_guard_pass'] is not False:raise ValueError('Unexpected result; independently review actual conclusion')
    seconds=sum(m['total_fit_seconds'] for m in optimizer['models'])
    artifacts=dict(run_id=RUN,status='ANALYZED',candidate='D92-SFHead-v1',candidate_promoted=False,goal_complete=False,
        release_commit='3348c8c37df402098a6f84397eaa1bf74d8d592e',records=2412,
        raw_scores=dict(local=str(result/'scores.json'),remote='/home/szu2070436088/2510044040/CV-SincNet/runs/'+RUN+'/scores.json',bytes=(result/'scores.json').stat().st_size),
        full_predictions='Remote each-row predictions.jsonl and sfhead/predictions.jsonl; all preserved',
        full_optimizer_trace='Remote each-row sfhead/fit_trace.jsonl; all1200 fits independently audited',
        numeric_audit='results/arithmetic_audit.json',optimizer_audit='results/optimizer_audit.json',
        summary='results/summary/report.md',compact_fit_logs='results/fit_logs/',
        source_data_access=False,new_ground_statistics_bytes=0,
        interpretation='Fails every-K H and new-accuracy improvement; no promotion or target-based retuning.')
    with (result/'artifacts.json').open('x',encoding='utf-8') as stream:json.dump(artifacts,stream,ensure_ascii=False,indent=2);stream.write('\n')
    extra=(f'\n\n## 完整核验与运行成本\n\n全部2412条评分已从混淆矩阵独立复算。1200次辅助拟合中，{optimizer["total_converged"]}次满足数值收敛条件，32次在K20达到预登记300次迭代上限；全部有限值，未回退或重跑。K1、K5、K10均收敛，仍未达到性能标准，因此不能把整体失败简单归为迭代次数不足。\n\n'
        f'累计实际拟合时间{seconds:.2f}秒，平均{seconds/1200:.3f}秒/次（N607 CPU环境，不能代表星载硬件速度）；不含特征提取、预测和日志序列化。实际FP64分类头数值状态为7728至33488字节，未将优化器临时内存混称为持久状态。新增地面统计传输0字节，Phase1模型未更新。\n\n'
        '该版本未晋级，整体优化目标未完成。四个模型seed的每K新类和H差值均为负；不筛选表现较好的新增类规模替代完整结论。确认集仅RX20-19，跨接收机结论有限。\n\n'
        '![各K的新旧类与H](old_new_h.png)\n\n![全部K与新增类规模的差值](delta_all_k_new.png)\n')
    summary_path=result/'summary/report.md'
    with summary_path.open('a',encoding='utf-8') as stream:stream.write(extra)
    report_path=folder/'report.md';report=report_path.read_text(encoding='utf-8')
    report=report.replace('尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。',
        '完整2412条结果已评分、独立核验并汇总。新类及H在所有K均下降，该候选未晋级，整体目标未完成。详见[完整结果](results/summary/report.md)与[产物定位](results/artifacts.json)。')
    report+='\n\n完整核验：2412条混淆矩阵复算通过；1200次辅助拟合完整日志核验，1168次收敛，32次K20到达固定迭代预算。两张图已目视检查。结果不回流调参或选择性重跑。\n'
    report_path.write_text(report,encoding='utf-8')
    evidence=folder/'evidence';evidence.mkdir(exist_ok=True)
    shutil.copyfile(WS/'local_artifacts/d92_upgrade_20260928/ground_payload_readback.json',evidence/'ground_payload_readback.json')
    print(json.dumps(artifacts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
