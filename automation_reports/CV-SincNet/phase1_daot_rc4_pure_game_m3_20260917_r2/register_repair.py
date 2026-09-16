"""Preserve failed r1; register identical scientific rows in a fresh r2 root."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent;old=ROOT.with_name(ROOT.name[:-1]+'1')
data=json.loads((old/'experiment.json').read_text(encoding='utf-8'))
data=json.loads(json.dumps(data,ensure_ascii=False).replace(old.name,ROOT.name))
data['replaces_run_id']=old.name;data['code']['commit']='REPAIR_COMMIT_IN_LANDING'
data['notes'].append('r1仅在训练前写resolved_config发生SatViewStage序列化错误；r2修复结构化序列化，科学配置/seed/预算不变。')
(ROOT/'experiment.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'events.jsonl').write_text(json.dumps(dict(at=datetime.now(timezone.utc).isoformat(),status='PLANNED',note='Same 21 rows; serialization-only repair, new root; no checkpoint inherited',evidence=['experiment.json',str(old/'launch_readback.json')]))+'\n',encoding='utf-8')
text=(old/'report.md').read_text(encoding='utf-8')
text+='\n## r2技术修复\n\nr1首行在训练前记录resolved_config时SatViewStage对象不能直接JSON序列化而退出。源域物理角色比对已通过，但未开始训练；dispatcher已退出，其余20行未启动。r1全部保留。r2仅将dataclass日程结构转换为JSON对象，保持所有科学参数、seed与预算不变。真实三seed配置回归：旧写法均复现同异常，新写法均通过无损JSON roundtrip。\n'
(ROOT/'report.md').write_text(text,encoding='utf-8')
for name in ('config_acceptance.json','local_acceptance.json','native_acceptance.json','review.md'):
    shutil.copy2(old/name,ROOT/name)
shutil.copytree(old/'resolved_configs',ROOT/'resolved_configs')
with (old/'report.md').open('a',encoding='utf-8') as f:f.write('\n## 首发状态：FAILED，已由r2替代\n\n首行训练前配置序列化失败；其余20行未启动，新调度器已退出。完整日志与产物保留。只修复记录序列化，用新的r2输出目录重新启动。\n')
print(ROOT)
