"""Truth-last source-only protected160 scoring on synthetic frozen predictions."""
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tests.test_pamf_phase3_star_protected_open_eval import fixture


def module():
    return importlib.import_module('experiments.pamf_phase3_open_world.star_protected_open_score')


def write_npz(path,data):
    with path.open('wb') as f: np.savez(f,**data)


def test_complete_predictions_and_calibration_precede_only_private_truth_and_primary_is_fixed(tmp_path):
    s=module()
    for arm in ('N0','R0','R1'):
        with fixture(tmp_path/arm,arm=arm) as (e,c,b,f,roles,events,truth):
            e.evaluate(c,'cpu'); prediction=Path(c['output_root']); calls=[];neighbors=[]
            nearest=s.geometry._nearest_indices
            def choose(*args):
                value=nearest(*args);neighbors.append(value);return value
            def private(path,novel):
                assert all((prediction/x).is_file() for x in ('complete.json','known_predictions.npz','novel_predictions.npz','source_aggregates.npz','thresholds.npy'))
                assert len(neighbors)==2 and all(set(value)==set(s.geometry.CONDITIONS) for value in neighbors)
                calls.append(path);return truth
            with patch.object(s.geometry,'_nearest_indices',choose),patch.object(s.common,'_truth',private):
                result=s.score(prediction,'synthetic-truth-only',tmp_path/arm/'score')
            assert calls==['synthetic-truth-only'] and len(result['source']['groups'])==42
            assert result['source']['overall']['A_K']==1 and result['source']['overall']['R_U']==1
            assert result['eligible_for_target_prereg']==(arm=='R1')
            assert result['A_N'] is None and result['H_OW'] is None and not result['final_phase3_complete']
            assert result['truth_last'] and not result['formal_target_accessed']
            assert result['all_geometry_neighbors_before_truth'] and result['source_geometry_diagnostic_only']
            assert result['optimizer_updates']==0 and set(result['source_geometry'])=={'original_fused160','protected160'}


def test_missing_or_corrupted_public_state_blocks_truth(tmp_path):
    s=module()
    for defect in ('incomplete','margin','identity','novel_y','aggregates','tau','features','smoke'):
        with fixture(tmp_path/defect) as (e,c,b,f,roles,events,truth):
            e.evaluate(c,'cpu');root=Path(c['output_root'])
            if defect=='incomplete': (root/'novel_predictions.npz').unlink()
            elif defect=='aggregates':
                data=dict(np.load(root/'source_aggregates.npz'));data['counts'][0,0,0]=0;write_npz(root/'source_aggregates.npz',data)
            elif defect=='tau': np.save(root/'thresholds.npy',np.zeros(6,np.float64)+123)
            elif defect=='features':
                data=dict(np.load(root/'novel_features.npz'));data['ids']=data['ids'][::-1];write_npz(root/'novel_features.npz',data)
            elif defect=='smoke':
                data=json.loads((root/'smoke.json').read_text());data['status']='FAIL';(root/'smoke.json').write_text(json.dumps(data))
            else:
                name='novel_predictions.npz' if defect=='novel_y' else 'known_predictions.npz'
                data=dict(np.load(root/name))
                if defect=='margin': data['scores'][0]+=1
                elif defect=='identity': data['baseline_closed'][0]=(data['baseline_closed'][0]+1)%6
                else: data['y']=np.zeros(len(data['ids']),np.int64)
                write_npz(root/name,data)
            with patch.object(s.common,'_truth') as private:
                with pytest.raises((ValueError,FileNotFoundError)): s.score(root,'forbidden',tmp_path/defect/'score')
                private.assert_not_called()


def test_known_calibration_must_reconstruct_before_truth(tmp_path):
    s=module()
    with fixture(tmp_path) as (e,c,b,f,roles,events,truth):
        e.evaluate(c,'cpu');root=Path(c['output_root']);path=root/'calibration/complete.json'
        report=json.loads(path.read_text());report['common_tail_rate']={'numerator':1,'denominator':2};path.write_text(json.dumps(report))
        with patch.object(s.common,'_truth') as private:
            with pytest.raises(ValueError): s.score(root,'forbidden',tmp_path/'score')
            private.assert_not_called()


def test_primary_unknown_failure_is_not_rescued_by_controls_and_missing_scenes_are_NA(tmp_path):
    s=module()
    with fixture(tmp_path,novel_accept=True) as (e,c,b,f,roles,events,truth):
        e.evaluate(c,'cpu');truth['scenario']=np.full(len(truth['ids']),e.SCENES[-1])
        with patch.object(s.common,'_truth',return_value=truth): result=s.score(c['output_root'],'synthetic',tmp_path/'score')
        assert not result['eligible_for_target_prereg'] and result['next_step']=='NO_PROMOTION'
        assert result['source']['known_42_groups_passed']
        assert result['source']['overall']['R_U']==0
        assert result['source']['by_scene'][e.SCENES[0]]['R_U'] is None
        assert not result['source']['by_scene'][e.SCENES[0]]['coverage']['unknown_complete']
