"""Preserve failed MVKME r01 and preregister one unchanged-formula technical recovery."""
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WS=Path('E:/type10-7')
OLD='20260928-phase2-d92-mvkme-repeat-rx3-m4-r01'
NEW='20260928-phase2-d92-mvkme-repeat-rx3-m4-r02'
OLD_RELEASE='d92_mvkme_repeat_rx3_20260928_r01'
NEW_RELEASE='d92_mvkme_repeat_rx3_20260928_r02'
OLD_CONFIG='configs/d92_mvkme_repeat_rx3_20260928.json'
NEW_CONFIG='configs/d92_mvkme_repeat_rx3_recovery_20260928.json'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def replaced(value):
    if isinstance(value,dict):return {k:replaced(v) for k,v in value.items()}
    if isinstance(value,list):return [replaced(v) for v in value]
    if isinstance(value,str):return value.replace(OLD,NEW).replace(OLD_RELEASE,NEW_RELEASE).replace(OLD_CONFIG,NEW_CONFIG).replace('configs/d92_mvkme_repeat_rx3_data_20260928.json','configs/d92_mvkme_repeat_rx3_recovery_data_20260928.json')
    return value


def main():
    if (ROOT/NEW_CONFIG).exists():raise FileExistsError('Recovery already prepared')
    failed=read(WS/'automation_reports/CV-SincNet'/OLD/'experiment.json')
    if failed['status']!='FAILED':raise ValueError('Prior failure must have terminal evidence')
    original=read(ROOT/OLD_CONFIG);recovery=replaced(original)
    recovery['replaces_run_id']=OLD
    recovery['description']+=' 技术恢复：修复既有Torch/NumPy双数组ABI的数值转换接口，使用Python数值列表中转；公式、精度、矩阵和预算不变。'
    recovery['notes'] += ['Prior r01 stopped before prediction/scoring; native synthetic smoke passed but array interoperability produced object dtype. Preserve all artifacts.',
        'One technical repair only: tensor/NumPy bridge uses numeric Python lists in both directions; no package/environment/model/formula change.',
        'Diagnostic evidence: local_artifacts/d92_upgrade_20260928/mvkme_numpy_bridge_diagnostic.json']
    assert recovery['confirmation']['candidate']==original['confirmation']['candidate']
    rx1_path=ROOT/'configs/d92_mvkme_repeat_rx1_20260928.json';rx1=read(rx1_path)
    record=read(WS/'automation_reports/CV-SincNet'/rx1['run_id']/'experiment.json')
    if record['status']!='PLANNED':raise ValueError('Do not change an already launched companion cohort')
    rx1['joint_benchmark']['cohort_run_ids']=recovery['joint_benchmark']['cohort_run_ids']
    rx1['notes'].append('Companion rx3 is technical-recovery r02; rx1 has never launched. Both use the same fixed formula and corrected numeric tensor bridge.')
    # Each release must own its unchanged capsule-reference configuration path.
    data_old=ROOT/'configs/d92_mvkme_repeat_rx3_data_20260928.json'
    data_new=ROOT/'configs/d92_mvkme_repeat_rx3_recovery_data_20260928.json'
    if data_new.exists():raise FileExistsError(data_new)
    recovery['confirmation']['data_config']=recovery['code']['cwd']+'/configs/'+data_new.name
    for path,value in [(ROOT/NEW_CONFIG,recovery),(data_new,read(data_old))]:
        with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
    rx1_path.write_text(json.dumps(rx1,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PREPARED_ONLY',run=NEW,companion=rx1['run_id'],formula_changed=False)))


if __name__=='__main__':main()
