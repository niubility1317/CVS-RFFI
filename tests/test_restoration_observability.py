import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
from experiments.pamf_restoration.observability import advance,periodicity


def test_fixed_delay_completion_uses_only_received_and_recovers_periodic_boundary():
    rng=np.random.default_rng(8)
    block=rng.normal(size=(2,16))+1j*rng.normal(size=(2,16))
    original=np.tile(block,(1,16))
    received=np.concatenate((np.zeros((2,24)),original[:,:232]),-1)
    result,period=advance(received,'periodic')
    assert np.allclose(result,original)
    assert set(period.tolist()) <= {16,32,64,128}
    assert np.allclose(advance(received,'zero')[0][:,:232],original[:,:232])
    assert periodicity(original)['16']['mean'] > .999999
