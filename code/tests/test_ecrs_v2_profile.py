import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.profile_ecrs_v2 import main


def test_synthetic_profile_records_scope_and_refuses_overwrite(tmp_path):
    path = tmp_path/'profile.json'
    args = ['--batch','1','--input-len','32','--warmup','0','--steps','1','--output',str(path)]
    main(args)
    before = path.read_bytes()
    result = json.loads(before)
    assert result['synthetic'] is True
    assert result['config']['input_shape'] == [1,2,32]
    assert 'backbones excluded' in result['full_inference_scope']
    assert len(result['crossfit']) == 2
    assert all(t['p50_ms'] >= 0 and t['timer'] == 'perf_counter' for t in result['timings'].values())
    with pytest.raises(SystemExit):
        main(args)
    assert path.read_bytes() == before
