import json
import shutil
from pathlib import Path

p = Path(__file__).resolve().parent
x = json.loads((p / 'user_status_latest.json').read_text(encoding='utf-8'))
s = json.loads((p / 'user_status_full_scan.json').read_text(encoding='utf-8'))
names = ['CORE90基线','grid基座','grid+X','grid+U','grid+原C2','grid+X+U，被动审计','grid+X+原C2','grid+U+原C2','grid+X+U+原C2','原生A1','原生A1+X','普通CORE90+原C2','grid+X+U+C*+暴露课程','grid+X+U+C*，固定课程','grid+X+U，仅暴露课程']
lines = ['# XUC15阶段数据核对：2026-09-13 09:40', '', '当前13/15组训练完成；M09完成109/200轮，M10完成114/200轮，均继续正常运行。整体状态TRAINING。测试预测及独立评分均未开始，当前没有目标域准确率、F1或混淆矩阵。', '', '调度器按已登记顺序等待15组训练全部结束，再固定全部预测，最后独立连接truth评分。本次未启动额外测试、未改参或重发。', '', '## 已完成训练数据', '', '下表全部200轮、9800次接受的主更新、seed392005。loss是E200的训练总损失，不是测试误差；不同目标项/课程的loss不宜直接用于性能排序。', '', '|实验|配置|E200 mean_loss|CORRECT次数|HOLD_CURRICULUM次数|训练小时|', '|---|---|---:|---:|---:|---:|']
for rid, row in s['rows'].items():
    if row['status'] != 'TRAINING_COMPLETE':
        continue
    art = x['artifacts'][rid]
    c = art['completion.json']['content']
    loss = art['logs.jsonl']['last']['mean_loss']
    lines.append(f"|{rid}|{names[int(rid[1:])]}|{loss:.4f}|{row['actions'].get('CORRECT',0)}|{row['actions'].get('HOLD_CURRICULUM',0)}|{c['elapsed_seconds']/3600:.2f}|")
lines += ['', '## 验证范围与机制边界', '', '完整解析13组共127400条actions记录和2600条epoch记录；另外完整解析M09/M10当前109/114条epoch记录及15组训练stdout。CORE90未接受主更新、记录中的非有限主loss/grad均为0；完整日志无Traceback、CUDA OOM或RC4非有限停止标记。native未提供skip计数，不能将缺失报告成0。', '', '原C2在M04/M06/M07/M08/M11自然触发CORRECT；所有CORE90组CATCHUP均为0。M04/M07各11次HOLD_CURRICULUM取自逐步actions，completion摘要将这些归入NORMAL，因此按详细日志列出。', '', 'M12/M13的C*控制器整个训练没有CATCHUP或CORRECT，记录CONTROL_NOT_ACTIVATED。当前不能声称主动C*控制或三机制协同收益已验证。M05本来只有被动审计，M14本来禁止控制动作，其无动作符合设计。', '', '## 测试数据可用性', '', '|场景|已完成预测组数|已有独立评分组数|准确率/F1|', '|---|---:|---:|---|', '|clean|0/15|0/15|尚未生成|', '|leo_clear_weak|0/15|0/15|尚未生成|', '|leo_low_elev_weak|0/15|0/15|尚未生成|', '|leo_rain_weak|0/15|0/15|尚未生成|', '', 'M09/M10每轮222步，CORE90每轮49步，原生对照训练预算约4.53倍；同为200轮不代表同等计算量。后续X在原生体系的受控比较应使用M10−M09。', '', '实时证据：[进程与产物快照](user_status_latest.json)、[完整日志扫描](user_status_full_scan.json)。之前最终checkpoint核验见同目录report.md；本次只读核对，不重复加载已验证模型。', '']
(p / 'user_status_20260913_0940.md').write_text('\n'.join(lines), encoding='utf-8')
dest = Path('E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/adv3b02-xuc-fusion-design-20260913/docs/research') / p.name
for name in ['user_status_latest.json','user_status_full_scan.json','user_status_20260913_0940.md','build_user_status.py']:
    shutil.copy2(p / name, dest / name)
print((p / 'user_status_20260913_0940.md').read_text(encoding='utf-8'))
