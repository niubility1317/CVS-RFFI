import json
import pytest
from experiments.pamf_phase2_practical.clean_mopc_full import verify_stability_evidence


def put(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')


def test_full_queue_rejects_missing_or_nonfinite_diagnostics(tmp_path):
    synthetic=tmp_path/'synthetic';real=tmp_path/'real'
    for gpu,seeds in ((5,(1731,1732)),(6,(1733,1734)),(7,(1735,1736))):
        put(synthetic/f'gpu{gpu}.log',[dict(scope='synthetic_only',semantics='public_code',seed=seed,
            query_read=False,truth_read=False,predictions_written=False,
            results=[dict(variant='no_clip',status='COMPLETE',completed_updates=160),
                     dict(variant='clip1',status='COMPLETE',completed_updates=160)]) for seed in seeds])
    for gpu,key,k,updates in ((5,'protonet_cda-s2026092702-c0195',20,560),
                             (6,'dadda_sda-s2026092703-c0332',10,320),
                             (7,'dadda_sda-s2026092703-c0332',20,560)):
        put(real/f'gpu{gpu}.log',[dict(event='SOURCE_SMOKE_PASS',query_rows=0,truth_read=False),
             dict(event='SAVED_B_SUPPORT_READY',key=key,k=k,n=20,query_rows=0,truth_read=False),
             dict(event='REGISTRATION_RETURNED',observed_nonfinite=False,completed_updates=updates)])
    assert verify_stability_evidence(synthetic,real)==9
    put(real/'gpu7.log',[dict(event='REGISTRATION_RETURNED',observed_nonfinite=True,completed_updates=560)])
    with pytest.raises(ValueError,match='diagnostic'):
        verify_stability_evidence(synthetic,real)
