"""Report the frozen full-FP32 candidate, without altering any predictions."""
import argparse
import json
from pathlib import Path
from experiments.cvs_equivariant_clean import analyze as original
from experiments.cvs_equivariant_clean.prepare_fp32 import RUN,SOURCE_RUN
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY


def analyze(root):
    evidence=root/'automation_reports/CV-SincNet'/RUN/'evidence'
    data=json.loads((evidence/'final_readback.json').read_text(encoding='utf-8'))
    new=[r for r in data['rows'] if r['resolved']['variant']=='equivariant_memory']
    if len(new)!=4 or any(r['resolved'].get('numerical_policy')!=FULL_FP32_POLICY or r['resolved'].get('backend_flags')!=FULL_FP32_POLICY for r in new):
        raise ValueError('All four frozen prediction numerical policies must match source')
    original.RUN=RUN;original.SOURCE_RUN=SOURCE_RUN;original.analyze(root)
    report=evidence.parent/'report.md';text=report.read_text(encoding='utf-8')
    text=text.replace('# CVS 全路径复相位等变记忆网络：独立 clean 测试报告','# CVS 整网复相位约束：完整 FP32 独立 clean 报告',1)
    text+='\n## 本轮数值策略\n\n本轮四个新模型从零训练，结构与参数数目保持不变；训练、源 V、公共物理诊断、profile 和 clean 预测均使用 cuDNN TF32=False、matmul TF32=False、benchmark=False、deterministic=False、matmul precision=highest。四份预测 actual flags 已与源 payload 一致核实。原20份控制预测只读复用，保留原历史精度，不宣称控制同为完整 FP32。\n\n本轮源策略在新 query 前依据源相位数值问题和公共冻结单变量证据提前固定；不存在从目标成绩到该策略的调整。数值容差在原冻结公共矩阵上的表现与实际识别性能分别报告，不能把关闭 TF32 自动等同于接收机因果分离或性能提升。\n'
    report.write_text(text,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);analyze(parser.parse_args().root)
