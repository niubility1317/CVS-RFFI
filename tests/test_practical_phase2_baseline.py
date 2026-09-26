import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_practical_phase2_baseline import source_provenance, factory, features, logits, ncm, author_adapt


def test_source_contract_and_target_contamination_rejected(tmp_path):
    expected = dict(role_ids={'L_s':['a'], 'U_s':['b'], 'V':['c']}, source_rxs=[1],
                    source_days=[1], ratios=[.07,.63,.3], split_seed=392005, num_classes=6)
    init = dict(scratch_only=True, checkpoint_sources=[], target_contact=False)
    complete = dict(target_evaluated=False, epoch=200, status='SOURCE_TRAINED')
    actual = dict(expected, physical_roles='EXACT_MATCH')
    def write():
        for filename, value in [('initialization.json',init),('completion.json',complete),('source_contract.json',actual)]:
            (tmp_path/filename).write_text(json.dumps(value),encoding='utf-8')
    write()
    source_provenance(tmp_path, expected)
    actual['role_ids'] = {'L_s':['other'], 'U_s':['b'], 'V':['c']}
    write()
    with pytest.raises(ValueError, match='CONTRACT_MISMATCH'):
        source_provenance(tmp_path, expected)
    actual['role_ids'] = expected['role_ids']
    init['target_contact'] = True
    write()
    with pytest.raises(ValueError, match='PROVENANCE'):
        source_provenance(tmp_path, expected)


@pytest.mark.parametrize('method',['cvcnn_ce','riei_fd','drift','poster','radionet'])
def test_query_is_sample_independent_and_has_all_classes(method):
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = factory(method,6,5).eval()
    x = torch.randn(4,2,256)
    before = {k:v.clone() for k,v in model.state_dict().items()}
    with torch.no_grad():
        z = features(model,method,x)
        score = logits(model,method,x)
        single = logits(model,method,x[:1])
    assert score.shape == (4,6)
    assert torch.allclose(score[:1],single,atol=2e-6,rtol=1e-5)
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    scores = ncm(z,torch.tensor([0,0,1,1]),z,2)
    assert scores.shape == (4,2) and torch.isfinite(scores).all()


@pytest.mark.parametrize('method',['poster','radionet'])
def test_author_adaptation_preserves_frozen_backbone_and_base(method):
    torch.set_num_threads(1)
    model = factory(method,6,5).eval()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    adapted = author_adapt(model,method,torch.randn(4,2,256).numpy(),torch.tensor([0,0,1,1]),2,9,'cpu')
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    prefix = ('conv1','conv2') if method=='poster' else ('blocks',)
    assert all(torch.equal(v,before[k]) for k,v in adapted.state_dict().items() if k.startswith(prefix))
    assert adapted(torch.randn(1,2,256)).shape == (1,2)


def test_scorer_waits_for_all_predictions_and_scores_old_new(tmp_path, monkeypatch):
    from tools.score_practical_phase2_baselines import main
    row = tmp_path/'row'
    row.mkdir()
    state = tmp_path/'state.json'
    state.write_text(json.dumps({'row':{'status':'RUNNING'}}))
    truth = tmp_path/'truth.json'
    monkeypatch.setattr(sys,'argv',['score','--run-root',str(tmp_path),'--truth',str(truth)])
    with pytest.raises(ValueError,match='Freeze all'):
        main()  # truth does not exist: gate must precede its opening
    state.write_text(json.dumps({'row':{'status':'PREDICTIONS_COMPLETE'}}))
    truth.write_text(json.dumps({'a':dict(pool_role='query',old=True,transmitter='old'),
                                 'b':dict(pool_role='query',old=False,transmitter='new')}))
    (row/'predictions_complete.json').write_text(json.dumps({'predictions':1}))
    (row/'predictions.jsonl').write_text(json.dumps(dict(query_ids=['a','b'],predicted_indices=[0,0],
        classes=['old','new'],split_id='s',mode='support_ncm',receiver='r',scenario='mid',k=1,support_seed=2))+'\n')
    main()
    result=json.loads((tmp_path/'scored_results.json').read_text())['results'][0]
    assert result['accuracy']==.5 and result['old_accuracy']==1 and result['new_accuracy']==0
    assert result['harmonic_mean']==0
