from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.dura_reference_report import report_complete_method
from experiments.pamf_ssl_research.hard_pl_early import RUN as EARLY_RUN,METHODS as EARLY
from experiments.pamf_ssl_research.hard_pl_guard import RUN as GUARD_RUN,METHODS as GUARD
from tests.test_pamf_hard_pl_cohort_report import fixture as hard_fixture
from tests.test_pamf_ssl_goal_counts import save


def fixture(root,run,methods,selected):
    rt=hard_fixture(root);rt.update(run_id=run,methods=list(methods))
    for row in rt['rows']:
        method=methods[0] if row['method']=='HPL2' else methods[1]
        row.update(method=method,row_id=method+'-s'+str(row['model_seed']))
        c=read(row['score_config']);c.update(run_id=run,row_id=row['row_id']);save(Path(row['score_config']),c)
        path=Path(c['output_root'])/'scores.json';v=read(path);v.update(run_id=run,row_id=row['row_id']);save(path,v)
    save(Path(rt['run_root'])/'state.json',dict(status='RUNNING',jobs={r['row_id']:dict(
        status='SCORED' if r['method']==selected else 'RUNNING') for r in rt['rows']}))
    return rt


CASES=[(run,methods,method) for run,methods in ((EARLY_RUN,EARLY),(GUARD_RUN,GUARD)) for method in methods]


@pytest.mark.parametrize('run,methods,selected',CASES)
def test_full_selected_three_seeds_report_without_any_pending_arm_files(run,methods,selected):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root,run,methods,selected)
        for row in rt['rows']:
            if row['method']!=selected:Path(row['score_config']).unlink()
        done=report_complete_method(rt,root/'report',selected);a=read(root/'report/analysis.json')
        scope=done['report_scope'];assert scope['models']==3 and scope['registered_full_models']==6
        assert not scope['report_is_full_run'] and not scope['full_registered_matrix_scored']
        assert scope['other_methods_pending']==[m for m in methods if m!=selected]
        assert {r['method'] for r in a['summary']}=={'F20','F21',selected}
        assert {r['readout'] for r in a['summary']}=={'raw','DURA-LU'}
        assert len(a['verdicts'])==2 and a['raw_goal_only']


@pytest.mark.parametrize('run,methods,selected',CASES)
def test_partial_selected_method_rejected_before_any_score_read(run,methods,selected):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root,run,methods,selected);path=Path(rt['run_root'])/'state.json';s=read(path)
        s['jobs'][selected+'-s2026092702']['status']='SCORING';save(path,s);opened=[]
        def track(p):opened.append(str(p));return read(p)
        with patch('experiments.pamf_ssl_research.dura_reference_report.read',side_effect=track):
            with pytest.raises(ValueError,match='before any score read'):report_complete_method(rt,root/'report',selected)
        assert opened==[str(path)] and not (root/'report').exists()


def test_complete_method_cannot_relabel_or_drop_registered_seed_rows():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root,GUARD_RUN,GUARD,GUARD[0])
        for method in ('HPL2',EARLY[0],'F21'):
            with pytest.raises(ValueError):report_complete_method(rt,root/'report',method)
        rt['rows'].pop()
        with pytest.raises(ValueError):report_complete_method(rt,root/'report',GUARD[0])
