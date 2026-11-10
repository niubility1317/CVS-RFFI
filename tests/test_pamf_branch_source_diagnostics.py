import json
import zlib
from experiments.pamf_branch_training.source_diagnostics import complete_steps, summarize_steps


def test_live_gzip_scan_keeps_only_complete_flushed_records(tmp_path):
    path = tmp_path / 'steps.gz'
    record = json.dumps({'metrics': {'branch_U_cross_anchors': 4, 'HPL_seen': 8}}).encode() + b'\n'
    encoder = zlib.compressobj(wbits=31)
    path.write_bytes(encoder.compress(record + b'{"metrics":') + encoder.flush(zlib.Z_SYNC_FLUSH))
    steps, closed = complete_steps(path)
    assert len(steps) == 1 and not closed
    summary = summarize_steps(steps)
    assert summary['branch_U_cross_anchors']['positive'] == 1
    assert summary['HPL_seen'] == 8
