import numpy as np
import pytest

from experiments.pamf_phase3_restoration.source_predicted_noise_rank import ranks, spearman, within_record, summarize


def test_rank_handles_ties_and_direction():
    np.testing.assert_array_equal(ranks([4,1,1,3]),[3,.5,.5,2])
    assert spearman([1,2,3],[3,2,1])==pytest.approx(-1.)
    assert spearman([1,1,1],[1,2,3]) is None
    with pytest.raises(ValueError):ranks([1,np.nan])


def test_same_record_extremes_exclude_ties():
    score=np.array([[0,4,2],[1,4,1],[2,4,0]],dtype=float)
    error=np.array([[0,9,0],[1,9,1],[2,9,2]],dtype=float)
    out=within_record(score,error)
    assert out['records']==2 and out['high_greater']==1 and out['high_lower']==1 and out['equal']==0
    with pytest.raises(ValueError):within_record(score[:2],error[:2])


def test_summarize_quartiles_cover_all_views():
    score=np.arange(1536,dtype=float).reshape(3,512)
    result=summarize(score,score*2,score*3,np.full((3,512),'clean'))
    assert result['spearman_predicted_vs_simulated_proxy']==pytest.approx(1.)
    assert sum(x['views'] for x in result['predicted_proxy_quartiles'])==1536
    assert result['same_record_max_view_agreement']=={'records':512,'matches':512}
