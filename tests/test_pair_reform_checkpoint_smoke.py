import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from post_stage_common import build_baseline_model


PATH = Path(__file__).resolve().parents[1]/'code/scripts/smoke_adv3b02_pair_reform.py'
SPEC = importlib.util.spec_from_file_location('pair_checkpoint_smoke', PATH)
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(2)
    args = dict(num_classes=3, num_domains=2, input_len=256, model_size='M',
                dataset='wisig', sample_rate_hz=25e6)
    model = build_baseline_model(SimpleNamespace(**args), torch.device('cpu'))
    path = tmp_path/'core90_shape_fixture.pth'
    torch.save({'model': model.state_dict(), 'baseline_args': args,
                'args': {'num_classes': 99}, 'epoch': 200}, path)
    return path


def test_checkpoint_smoke_all_six_paths_and_step(checkpoint, tmp_path):
    output = tmp_path/'smoke.json'
    assert SMOKE.main(['--checkpoint', str(checkpoint), '--output-json', str(output), '--device', 'cpu']) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['status'] == 'VERIFIED'
    assert result['query_inputs'] == result['target_inputs'] == 0
    assert result['checkpoint_compatibility']['missing'] == []
    assert result['checkpoint_compatibility']['unexpected'] == []
    assert result['teacher_domain_forwards'] == 0
    assert set(result['rows']) == {'point','safe','asymmetric','tangent_only','route_only','memory'}
    assert all(row['optimizer_steps'] >= 1 and row['parameters_changed'] for row in result['rows'].values())
    assert result['rows']['memory']['memory_hit_rate'] == 1.


def test_reject_incompatible_checkpoint_without_silent_head_injection(checkpoint):
    payload = torch.load(checkpoint, weights_only=False)
    payload['model']['unexpected_fake_head.weight'] = torch.zeros(1)
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match='compatibility'):
        SMOKE.load_model(checkpoint, torch.device('cpu'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA AMP')
def test_checkpoint_smoke_cuda_amp_real_namespace_l_and_u(checkpoint):
    result=SMOKE.run_smoke(checkpoint,torch.device('cuda'),amp=True)
    assert result['amp'] and result['status']=='VERIFIED'
    for row in result['rows'].values():
        assert [s['role'] for s in row['steps']]==['L','U']
        assert all(s['amp'] for s in row['steps'])
        assert 'orbit_logit' in row['steps'][1]['components']
