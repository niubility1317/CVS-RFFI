import json
import pytest

from experiments.pamf_phase2_practical.r42_direct_c_readout_wait import wait


def test_wait_refuses_to_analyze_after_failed_run(tmp_path):
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'run_root': str(tmp_path)}), encoding='utf-8')
    (tmp_path / 'failed.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='worker failed'):
        wait(str(runtime), str(tmp_path / 'native.csv'), str(tmp_path / 'out'), 12345)
    assert not (tmp_path / 'out').exists()
