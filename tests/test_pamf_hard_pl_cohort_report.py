from pathlib import Path
import tempfile
from unittest.mock import patch
import pytest
from experiments.cric_dura_comparison.io import read
from experiments.pamf_ssl_research.hard_pl import RUN,METHODS,RESOURCE_GPU_BY_SEED
from experiments.pamf_ssl_research.dura_reference_report import report,report_complete_hpl2
from tests.test_pamf_ssl_dura_reference import report_fixture
from tests.test_pamf_ssl_goal_counts import save


def fixture(root):
    rt=report_fixture(root);rt.update(run_id=RUN,methods=list(METHODS))
    rt['rows']=[r for r in rt['rows'] if r['method'] in ('PSM','RXPSM')]
    for row in rt['rows']:
        row.update(method='HPL2' if row['method']=='PSM' else 'HPLT1U')
        row.update(row_id=row['method']+'-s'+str(row['model_seed']),gpu=RESOURCE_GPU_BY_SEED[row['model_seed']])
        c=read(row['score_config']);c.update(run_id=RUN,row_id=row['row_id']);save(Path(row['score_config']),c)
        path=Path(c['output_root'])/'scores.json';s=read(path);s.update(run_id=RUN,row_id=row['row_id']);save(path,s)
    save(Path(rt['run_root'])/'state.json',dict(status='RUNNING',jobs={r['row_id']:dict(
        status='SCORED' if r['method']=='HPL2' else 'RUNNING') for r in rt['rows']}))
    return rt


def test_only_full_registered_HPL2_cohort_all_seeds_both_readouts_and_goal():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root)
        # Pending HPLT1U target files must never be opened.
        for row in rt['rows']:
            if row['method']=='HPLT1U':Path(row['score_config']).unlink()
        done=report_complete_hpl2(rt,root/'report');a=read(root/'report/analysis.json')
        assert done['status']=='ANALYZED_COMPLETE_METHOD' and not done['report_scope']['report_is_full_run']
        assert done['report_scope']['other_methods_pending']==['HPLT1U'] and not done['report_scope']['full_registered_matrix_scored']
        assert {r['method'] for r in a['summary']}=={'F20','F21','HPL2'}
        assert {r['readout'] for r in a['summary']}=={'raw','DURA-LU'}
        assert len([r for r in a['summary'] if r['method']=='HPL2'])==16
        assert a['model_seeds']==[2026092701,2026092702,2026092703] and a['raw_goal_only']


def test_unfinished_HPL2_row_rejected_before_any_target_score_open():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root);state=read(Path(rt['run_root'])/'state.json')
        state['jobs']['HPL2-s2026092702']['status']='SCORING';save(Path(rt['run_root'])/'state.json',state)
        calls=[]
        def tracked(path):
            calls.append(str(path));return read(path)
        with patch('experiments.pamf_ssl_research.dura_reference_report.read',side_effect=tracked):
            with pytest.raises(ValueError,match='before any score read'):report_complete_hpl2(rt,root/'report')
        assert len(calls)==1 and calls[0].endswith('state.json')


def test_full_experiment_report_still_requires_original_all_scored_matrix():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root)
        with pytest.raises(ValueError,match='Complete supplemental matrix'):report(rt,root/'report')


def test_incomplete_or_relabelled_experiment_cannot_use_cohort_reporting():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);rt=fixture(root);rt['rows'].pop()
        with pytest.raises(ValueError):report_complete_hpl2(rt,root/'report')
        rt=fixture(root);rt['run_id']='other'
        with pytest.raises(ValueError):report_complete_hpl2(rt,root/'report')
