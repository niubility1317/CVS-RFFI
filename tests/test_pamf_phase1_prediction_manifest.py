import pytest

from experiments.pamf_phase1_flow import experiment as core


def manifest():
    return dict(status='PREDICTIONS_COMPLETE',count=10,views=list(core.VIEWS),
                F_run=core.F_RUN,F_row=core.F_ROW,commit='fixed-commit',TX_truth_reads=0,model_updates=0)


def test_scorer_accepts_fixed_predictions_with_valid_accepted_counts():
    done=manifest()
    done['accepted_by_view']={view: 0 for view in core.VIEWS}
    done['accepted_by_view']['practical_high']=5
    core.validate_prediction_manifest(done,count=10,commit='fixed-commit')


@pytest.mark.parametrize('change', [
    lambda d:d.update(TX_truth_reads=1),
    lambda d:d.update(model_updates=1),
    lambda d:d.update(accepted_by_view={'clean': 0}),
    lambda d:d.update(accepted_by_view={view: 11 for view in core.VIEWS}),
    lambda d:d.update(unexpected='not part of the contract'),
])
def test_scorer_rejects_incomplete_or_inconsistent_manifest(change):
    done=manifest();change(done)
    with pytest.raises(ValueError):
        core.validate_prediction_manifest(done,count=10,commit='fixed-commit')
