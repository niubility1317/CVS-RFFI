"""Copy this task's explicit new files and registration; preserve other work."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=Path('E:/type10-7')
RUN='20260928-diagnostic-d92-scv-source-s2026092701-r01'
FILES=['code/cvsrffi/stage2_d92_support_cv.py','tests/test_d92_support_cv.py',
    'tools/export_d92_source_development.py','tools/evaluate_d92_source_development.py',
    'tools/prepare_d92_source_development.py','tools/run_d92_source_development.py',
    'tools/publish_d92_source_development.py','configs/d92_upgrade_source_20260928.json',
    'docs/D92_SUPPORT_UPGRADE_20260928.md']


def main():
    for relative in FILES:
        source=ROOT/relative;target=WORKSPACE/relative
        text=source.read_text(encoding='utf-8')
        if '\ufffd' in text or text.startswith('\ufeff'):raise ValueError('Encoding error '+relative)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if target.read_bytes()!=source.read_bytes():raise ValueError('Mirror mismatch')
    report=WORKSPACE/'automation_reports/CV-SincNet'/RUN
    shutil.copyfile(ROOT/'configs/d92_upgrade_source_20260928.json',report/'experiment.json')
    text=(report/'report.md').read_text(encoding='utf-8')
    if '## 实施与审查' not in text:
        with (report/'report.md').open('a',encoding='utf-8',newline='\n') as f:
            f.write('\n## 实施与审查\n\n当前状态LOCAL_VERIFIED。完整目标与机制见docs/D92_SUPPORT_UPGRADE_20260928.md；新候选8项聚焦测试通过，唯一P0/P1审查完成。源域仅旧6类，H不适用；不得声称目标性能提升。原D92和所有历史输出保持不变。精确运行命令、commit、PID与状态在启动后读回。\n')
    for name in ['experiment.json','report.md','events.jsonl']:
        target=ROOT/'automation_reports/CV-SincNet'/RUN/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(report/name,target)
    # Only the new run's small index entry; full workspace index remains at its
    # existing canonical location and is not copied over another branch's index.
    catalog=ROOT/'experiment_registry/d92_upgrade_20260928.jsonl'
    catalog.parent.mkdir(parents=True,exist_ok=True)
    catalog.write_text(json.dumps(dict(run_id=RUN,report='automation_reports/CV-SincNet/'+RUN+'/report.md',
        experiment='automation_reports/CV-SincNet/'+RUN+'/experiment.json',status='LOCAL_VERIFIED'))+'\n',encoding='utf-8')
    print('UTF8_AND_EXPLICIT_MIRRORS_VERIFIED')


if __name__=='__main__':main()
