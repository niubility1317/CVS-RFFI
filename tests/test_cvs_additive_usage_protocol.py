import copy,json
from pathlib import Path
import pytest
import torch
from experiments.cvs_additive_usage import audit,publish


def config():return json.loads((Path(__file__).resolve().parents[1]/'experiments/cvs_additive_usage/configs/source.json').read_text(encoding='utf-8'))


def test_actual40row_configuration_and_publisher_scripts():
    c=config();audit.validate_config(c)
    spec=json.loads((Path(__file__).resolve().parents[1]/'experiments/cvs_additive_usage/configs/experiment_spec.json').read_text(encoding='utf-8'))
    rows=spec['rows'];assert len(rows)==40 and len({r['output_root'] for r in rows})==40
    assert {(r['variant'],r['seeds']['model'],r['internal_mode']) for r in rows}=={(v,s,m) for v in audit.VARIANTS for s in audit.SEEDS for m in audit.MODES}
    assert all(r['optimizer'] is None and r['epochs'] is None for r in rows)
    assert all(r['checkpoint'].startswith(audit.SOURCE+'/') and r['source_commit']==audit.SOURCE_COMMIT for r in rows)
    compile(publish.REMOTE.replace('CONFIG',repr(dict(project='p',run='r',release='s',config='c',python='py',sha256='h',archive='a',commit='t'))),'remote','exec')
    compile(publish.INSPECT.replace('CONFIG',repr(dict(project='p',run='r',release='s'))),'inspect','exec')
    assert 'experiments/cvs_synchronized_identity/' in publish.PREFIXES


@pytest.mark.parametrize('key',['target_truth','p1_capsule','teacher','resume','augmentation'])
def test_overrides_and_scope_changes_rejected(key):
    c=config();c[key]=True
    with pytest.raises(ValueError):audit.validate_config(c)


def test_modes_budget_and_selection_remain_fixed():
    for change in [dict(modes=['trained']),dict(optimizer_steps=1),dict(used_for_selection=True),dict(model_seeds=[2026092701])]:
        with pytest.raises(ValueError):audit.validate_config(dict(config(),**change))


@pytest.mark.parametrize('key',['epoch','source_contract','initialization','method','config','classes'])
def test_checkpoint_payload_header_corruption_rejected(key):
    row=dict(variant='additive_equivariant');resolved=dict(model_seed=2026092701,backend_flags=audit.FULL_FP32_POLICY)
    contract=dict(classes=['t'+str(i) for i in range(6)],role_ids={'L_s':['l'],'V':['v']});initial=dict(status='SCRATCH',ancestors=[])
    payload=dict(epoch=200,source_contract=contract,initialization=initial,selection='fixed_last_epoch',method='cvs_additive_identity',variant=row['variant'],config=resolved,classes=contract['classes'],num_classes=6)
    audit.validate_payload(payload,row,resolved,contract,initial)
    bad=copy.deepcopy(payload);bad[key]='corrupted'
    with pytest.raises(ValueError):audit.validate_payload(bad,row,resolved,contract,initial)


def test_source_accuracy_uses_exact_counts_not_float32_rounding():
    cm=torch.zeros(6,6,dtype=torch.int64);cm[0,0]=26273;cm[0,1]=727
    result=audit.metrics(cm,27000.,727,.2)
    assert result['accuracy']==26273/27000 and sum(map(sum,result['confusion']))==27000
