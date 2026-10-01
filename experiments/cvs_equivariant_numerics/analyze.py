"""Analyze all preregistered frozen/public numerical rows; no target data paths."""
import argparse
import csv
import json
import math
from pathlib import Path
from experiments.cvs_equivariant_numerics.audit import SEEDS, MODES, PHASES, BASE_COMMIT


def analyze(evidence):
    data=json.loads((evidence/'readback.json').read_text(encoding='utf-8'))
    rows=data['rows']
    assert not data['process'] and data['completion']['status']=='DIAGNOSTIC_MATRIX_COMPLETE'
    assert len(rows)==12 and {(r['seed'],r['mode']) for r in rows}=={(s,m) for s in SEEDS for m in MODES}
    assert data['resolved']['model_code_commit']==BASE_COMMIT
    assert data['completion']['backend_flags_restored'] and not data['completion']['target_access']
    assert not data['completion']['model_updated']
    initial=data['resolved']['initial_flags']
    flat=[];stages={};unchanged_original=True
    for row in rows:
        assert row['state_unchanged'] and not row['model_updated'] and not row['target_access']
        wanted=dict(initial,cudnn_allow_tf32=initial['cudnn_allow_tf32'] if row['mode']=='default_fp32' else False)
        assert row['flags']==wanted
        if row['status']!='DIAGNOSTIC_COMPLETE':
            flat.append(dict(seed=row['seed'],mode=row['mode'],status=row['status'],logit_max_error=None,unit_embedding_max_distance=None,synthetic_argmax_changes=None,original_exact_reproduction=None))
            continue
        physical=row['original_audit'];phase=physical['phase_audit']
        assert len(physical['records'])==30 and tuple(p['theta_radians'] for p in phase)==PHASES
        assert not physical['target_access'] and not physical['model_updated'] and not physical['formal_data_access']
        exact=phase==row['original_source_phase_audit'] if row['mode']=='default_fp32' else None
        if exact is False:unchanged_original=False
        err=max(p['whole_logit_max_abs_error'] for p in phase)
        distance=max(p['whole_unit_embedding_max_distance'] for p in phase)
        assert all(math.isfinite(v) for v in (err,distance))
        assert len(row['stage_audit'])==3
        changes=sum(p['synthetic_argmax_change_count'] for p in row['stage_audit'])
        flat.append(dict(seed=row['seed'],mode=row['mode'],status=row['status'],logit_max_error=err,unit_embedding_max_distance=distance,synthetic_argmax_changes=changes,original_exact_reproduction=exact))
        for p in row['stage_audit']:
            for stage in p['stages']:
                key=(row['mode'],stage['stage'])
                stages[key]=max(stages.get(key,0.),stage['max_abs_error'])
    summary={m:dict(logit_max_error=max(r['logit_max_error'] for r in flat if r['mode']==m and r['logit_max_error'] is not None),
        unit_embedding_max_distance=max(r['unit_embedding_max_distance'] for r in flat if r['mode']==m and r['unit_embedding_max_distance'] is not None),
        synthetic_argmax_changes=sum(r['synthetic_argmax_changes'] or 0 for r in flat if r['mode']==m)) for m in MODES}
    for name, records in [('row_summary.csv',flat),('stage_summary.csv',[dict(mode=m,stage=s,max_abs_error=v) for (m,s),v in stages.items()])]:
        with (evidence/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    result=dict(status='VERIFIED',all12_rows_complete=data['completion']['complete_rows']==12,original_default_phase_exact_reproduction=unchanged_original,
        modes=summary,source_only=True,target_access=False,model_updated=False,
        precision_scope='RequestedFP64 preserves native Sinc/headFP32;not end-to-endFP64',
        target_regression_cause='UNPROVEN: no target access or attribution experiment',
        goal_achieved=False)
    (evidence/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=evidence.parent/'report.md'
    text=report.read_text(encoding='utf-8')+'\n## 完成结果\n\n状态 ANALYZED，远端自然结束，12 行均有完整 result；固定模型 state_dict 不变、进程标志恢复，未访问正式 IQ 或目标。\n\n| 模型 seed | 精度模式 | 最大 logit 误差 | 最大单位嵌入距离 | 合成 argmax 变化 |\n|---|---|---:|---:|---:|\n'
    for row in flat:text+=f"| {row['seed']} | {row['mode']} | {row['logit_max_error']:.10g} | {row['unit_embedding_max_distance']:.10g} | {row['synthetic_argmax_changes']} |\n"
    text+='\n原默认 FP32 的 4seed×3phase 与原 source 物理诊断逐值完全一致：'+str(unchanged_original)+'。第三模式保留原生 Sinc/head 的 FP32 运算，不能称为整网 FP64。合成 argmax 只是相位一致性，合成设置没有真实身份标签。\n\n| 模式 | 全部 seed 最大 logit 误差 | 全部 seed 最大单位嵌入距离 |\n|---|---:|---:|\n'
    for mode,m in summary.items():text+=f"| {mode} | {m['logit_max_error']:.10g} | {m['unit_embedding_max_distance']:.10g} |\n"
    text+='\n[完整远端读回](evidence/readback.json) · [逐行汇总](evidence/row_summary.csv) · [逐层汇总](evidence/stage_summary.csv) · [分析核实](evidence/analysis.json)。\n'
    report.write_text(text,encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    analyze(parser.parse_args().evidence)
