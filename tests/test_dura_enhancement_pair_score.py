import numpy as np
import pytest
from experiments.pamf_dura_efficiency.enhancement_pair_score import statistics,correction_metrics


def test_pair_repairs_and_damage_preserve_correct_count_difference():
    labels=np.array([0,1,2,3,4,5]);before=np.array([0,0,2,1,4,3]);after=np.array([1,1,2,3,4,0])
    value=statistics(labels,after,before)
    assert (value['repaired'],value['damaged'],value['net_repaired'],value['predictions_changed'])==(2,1,1,4)
    assert value['correct']==4 and value['count']==6


def test_actual_correction_bound_and_gain_phase_checks():
    rng=np.random.default_rng(24);base=rng.normal(size=(4,2,256))
    z=base[:,0]+1j*base[:,1];v=rng.normal(size=z.shape)+1j*rng.normal(size=z.shape)
    v-=z*((z.conj()*v).sum(1)/(abs(z)**2).sum(1))[:,None]
    delta=np.stack((v.real,v.imag),1)*.01
    value=correction_metrics(base,base+delta)
    assert value['maximum_relative_RMS']<.05 and value['maximum_energetic_complex_orthogonality']<1e-12
    with pytest.raises(ValueError,match='five-percent'):correction_metrics(base,base+10*delta)
    with pytest.raises(ValueError,match='gain/phase'):correction_metrics(base,base*1.01)
