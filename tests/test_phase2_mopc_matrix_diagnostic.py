import json
import pytest

from experiments.pamf_phase2_practical.clean_mopc_matrix_diagnostic import cases, diagnostic_result


def test_cases_require_exact_original_mopc_points_and_unique_identity(tmp_path):
    job=tmp_path/'job.json'
    job.write_text(json.dumps(dict(profile='dadda_sda',methods=['mopc_hr'],points=[[1,2],[5,20]])))
    source=dict(group='hybrid',job=str(job),runtime='runtime.json')
    assert len(cases(dict(sources=[source],points=[[1,2],[5,20]])))==2
    with pytest.raises(ValueError,match='Duplicate'):
        cases(dict(sources=[source,source],points=[[1,2]]))
    with pytest.raises(ValueError,match='point'):
        cases(dict(sources=[source],points=[[10,20]]))


def test_diagnostic_result_rejects_query_or_failed_process(tmp_path):
    file=tmp_path/'log';result=dict(event='DIAGNOSTIC_RESULT',status='COMPLETE',query_read=False,
                                    truth_read=False,predictions_written=False)
    file.write_text(json.dumps(result)+'\n')
    assert diagnostic_result(file,0)['status']=='COMPLETE'
    with pytest.raises(ValueError):diagnostic_result(file,1)
    file.write_text(json.dumps(dict(result,query_read=True))+'\n')
    with pytest.raises(ValueError):diagnostic_result(file,0)
