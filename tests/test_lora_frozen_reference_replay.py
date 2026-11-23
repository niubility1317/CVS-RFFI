from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.cric_dura_comparison.io import write
from experiments.pamf_lora_main.residual_reference_replay import (
    GROUPS, SEEDS, TARGET, completion, index_for, score,
)


def test_receiver_shards_partition_single_frozen_matrix():
    seen = [rx for receivers in GROUPS.values() for rx in receivers]
    assert sorted(seen) == sorted(TARGET)
    assert len(index_for(SEEDS, seen)) == 420


def test_missing_frozen_files_keep_truth_archive_closed(tmp_path):
    out, old = (tmp_path / 'new'), (tmp_path / 'old')
    write(out / 'predictions' / 'complete.json', completion())
    write(old / 'predictions' / 'complete.json', dict(status='PARTIAL'))
    with patch('zipfile.ZipFile') as open_truth:
        with pytest.raises(ValueError, match='Truth remains closed'):
            score(out, old, Path('unused-data-path'))
        open_truth.assert_not_called()
