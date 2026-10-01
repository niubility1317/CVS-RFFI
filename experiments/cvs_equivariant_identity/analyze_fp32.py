"""Complete source report for the new full-FP32 run; no target score inputs."""
import argparse
import json
from pathlib import Path
from experiments.cvs_equivariant_identity import analyze as original
from experiments.cvs_equivariant_identity.prepare_fp32 import RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def analyze(root):
    evidence=root/'automation_reports/CV-SincNet'/RUN/'evidence'
    data=json.loads((evidence/'source_research_complete.json').read_text(encoding='utf-8'))
    for row in data['rows']:
        if row['resolved'].get('numerical_policy')!=FULL_FP32_POLICY or row['resolved'].get('backend_flags')!=FULL_FP32_POLICY or row['completion'].get('backend_flags')!=FULL_FP32_POLICY:
            raise ValueError('Actual full-FP32 source policy evidence differs')
        if any(epoch.get('cudnn_allow_tf32') is not False for epoch in row['epochs']):
            raise ValueError('Full source epoch numerical state differs')
    original.RUN=RUN;original.analyze(root)
    report=evidence.parent/'report.md';text=report.read_text(encoding='utf-8')
    text=text.replace('# CVS 全路径复相位等变记忆网络：完整源实验报告','# CVS 整网复相位约束：完整 FP32 源实验报告',1)
    text+='\n## 实际精度策略\n\n新模型从零训练，结构/202553 参数和预算不变；训练、V 验证、公共物理诊断、实测 profile 均在 cuDNN.allow_tf32=False 的同一上下文内。实际 matmul TF32=False、benchmark=False、deterministic=False、matmul precision=highest 与固定策略一致，40000 步和 800 轮实际 TF32 状态由完整日志审计核实。原 residual 源控制保留其历史精度，本轮比较不声称控制同为完整 FP32。\n\n若新候选被固定源规则选中，clean 预测继承源 payload 的实际数值策略，在首次 forward 前生效并读回；已有目标预测不修改、不重测，目标成绩不回流。该策略源于新 query 前的源相位问题和公共冻结单变量诊断，尚不能直接证明识别性能提高。\n'
    report.write_text(text,encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root)
