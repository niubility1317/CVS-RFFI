"""Focused snapshot and truth-barrier checks without query data."""
from unittest.mock import patch
from experiments.cvs_state_test_now import design as d,evaluate as e

def checks():
    rows=d.rows()
    assert len(rows)==24 and len({c['row_id'] for c in rows})==24
    assert all(d.config(c)==c and c['run_id']==d.parent.RUN and c['output_root'].startswith(d.parent.BASE.as_posix()) for c in rows)
    value=dict(status='ALL_SOURCE_FROZEN',run_id=d.RUN,rows=rows,target_access=False,selection='ALL_COMPLETED_AT_FIXED_SNAPSHOT')
    with patch.object(e,'read',return_value=value):assert e.snapshot(d)==rows
    for key,bad in [('rows',rows[:-1]),('target_access',True),('selection','ranked_by_test')]:
        with patch.object(e,'read',return_value=dict(value,**{key:bad})):
            try:e.snapshot(d)
            except ValueError:pass
            else:raise AssertionError('Changed freeze accepted')
    with patch.object(e,'snapshot',return_value=rows),patch.object(e,'prediction_preflight',side_effect=ValueError('incomplete')),patch.object(e,'read') as read:
        try:e.score()
        except ValueError:pass
        else:raise AssertionError('Incomplete predictions reached scorer')
        read.assert_not_called()
    results=[dict(arm=c['arm'],model_seed=c['model_seed'],view=v,dimension='overall',accuracy=.8,macro_f1=.79) for c in rows for v in e.VIEWS]
    pairs=e.paired_results(results)
    assert pairs and all(p['mean_difference']==0 for p in pairs)
    assert all(p['treatment'] in d.ARMS and p['control'] in d.ARMS for p in pairs)
    assert e.BASE==d.BASE and e.BASE!=d.parent.BASE
    from experiments.cvs_state_test_now import views
    assert views.ROOT==d.BASE/'weak_views' and e.WEAK_ROOT==views.ROOT
    print('PASS: 24 immutable source configs; changed freeze rejected; truth barrier; partial-arm pairs; isolated outputs')

if __name__=='__main__':checks()
