import numpy as np
from experiments.phase1_dura_orthogonal.source_mix_frozen_agreement import paired


def test_paired_gains_and_losses_against_same_clean_proxy():
    result=paired(np.array([0,1,2,3]),np.array([0,0,2,0]),np.array([1,1,2,3]))
    assert result=={'gains':2,'losses':1}
