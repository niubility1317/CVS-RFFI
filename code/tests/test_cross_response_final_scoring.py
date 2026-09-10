import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cvsrffi.cross_response.final_scoring import (
    SCENARIOS, ReceivedRecord, write_fixed_predictions, score_fixed_predictions,
    wisig_received_catalog, evaluate_final_truth_last,
)


class IQClassifier(nn.Module):
    num_classes = 2

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.))
        self.calls = 0

    def forward(self, x, *, y_tx, domain_labels, return_aux):
        assert y_tx is None and domain_labels is None and not return_aux
        assert not self.training
        self.calls += 1
        value = x[:,0].mean(-1)*self.weight
        return torch.stack([value,-value],dim=-1)


def args():
    return SimpleNamespace(num_classes=2,eval_batch_size=2,eval_max_batches=0,sat_eval_max_batches=-1,
                           sat_seed=42,sat_fs_hz=25e6,sat_fc_hz=2.462e9)


def fixture_records():
    return [ReceivedRecord(f'opaque_{i}',lambda i=i: torch.stack([
        torch.ones(256)*(1 if i%2==0 else -1),torch.zeros(256)]),receiver=i%2,day=0)
        for i in range(4)]


def identity_channel(iq, scenario, args, *, gen, return_meta):
    assert scenario in SCENARIOS[1:] and not return_meta
    return iq, None


def test_all_four_complete_scenes_precede_first_truth_access(tmp_path):
    directory = tmp_path/'fixed'
    model = IQClassifier()
    records = fixture_records()
    ids = [r.record_id for r in records]
    events = []
    def truth(expected):
        assert model.calls == 8
        assert (directory/'prediction_manifest.json').is_file()
        for scenario in SCENARIOS:
            rows = (directory/f'predictions.{scenario}.jsonl').read_text().splitlines()
            assert len(rows) == 4
            assert all('label' not in json.loads(line) for line in rows)
        assert not (directory/'scoring_truth.jsonl').exists()
        events.append('truth_read')
        return {rid:i%2 for i,rid in enumerate(expected)}
    write_fixed_predictions(model,records,{'test_seen_day_unseen_rx':ids,'test_unseen_day_unseen_rx':ids},args(),torch.device('cpu'),
                            directory,channel_apply=identity_channel)
    assert events == [] and model.training
    report = score_fixed_predictions(directory,truth)
    assert events == ['truth_read'] and report['truth_last']
    assert report['prediction_rows'] == 16 and report['test']['tx_acc'] == 100
    assert report['test']['tx_total'] == 4  # Aliased main groups cannot double-count physical records.
    assert set(report['sat_test_named']) == set(SCENARIOS[1:])
    for scene in report['sat_test_named'].values():
        assert scene['aggregate']['tx_total'] == 4
    disk_report = json.loads((directory/'independent_scores.json').read_text(),
                             parse_constant=lambda value: pytest.fail(f'nonstandard JSON number:{value}'))
    assert disk_report['named_test']['test_seen_day_unseen_rx']['dom_acc'] is None


def test_real_training_frozen_evaluation_routes_to_truth_last(tmp_path,monkeypatch):
    from SSDG import train_ssdg as train
    from cvsrffi.cross_response import final_scoring
    model=IQClassifier()
    path=tmp_path/'final.pth'
    path.write_bytes(b'fixture-checkpoint')
    current=args()
    current._cross_response_resolved={'enabled':False}
    current.checkpoint_selection='final_only'
    checkpoint={'epoch':200,'stats':{'val':{}}}
    monkeypatch.setattr(train,'load_checkpoint',lambda *a: checkpoint)
    monkeypatch.setattr(train,'_validate_phase1_checkpoint_payload',lambda p,*a:p)
    monkeypatch.setattr(train,'_load_phase1_checkpoint_strict',lambda *a:None)
    def forbidden(*a,**kw):
        raise AssertionError('legacy online truth evaluation entered')
    monkeypatch.setattr(train,'evaluate_named_loaders',forbidden)
    monkeypatch.setattr(train,'_evaluate_sat_if_enabled',forbidden)
    monkeypatch.setattr(train,'protected_metric_snapshot',lambda **kw:{'checked':True})
    calls=[]
    def evaluate(m,a,ctx,device,out):
        calls.append(out)
        assert m is model and a is current
        return {'status':'COMPLETE','truth_last':True,'test':{},'named_test':{},'sat_test_named':{}}
    monkeypatch.setattr(final_scoring,'evaluate_final_truth_last',evaluate)
    result=train._run_final_heldout_evaluation(current,model,{},torch.device('cpu'),path)
    assert calls==[tmp_path] and result['truth_last']
    assert result['checkpoint_epoch']==200 and result['protected_metrics']=={'checked':True}


@pytest.mark.parametrize('corruption',['missing_record','duplicate_record','missing_scene','bad_class_width'])
def test_incomplete_predictions_rejected_without_truth_access(tmp_path,corruption):
    directory = tmp_path/'fixed'
    records = fixture_records()
    write_fixed_predictions(IQClassifier(),records,{'test':[r.record_id for r in records]},args(),
                            torch.device('cpu'),directory,channel_apply=identity_channel)
    path = directory/'predictions.leo_rain_weak.jsonl'
    rows = path.read_text().splitlines()
    if corruption == 'missing_scene':
        path.unlink()
    else:
        if corruption == 'missing_record':
            rows.pop()
        elif corruption == 'duplicate_record':
            rows.append(rows[0])
        else:
            row = json.loads(rows[0]); row['logits'] = [1.]; rows[0] = json.dumps(row)
        path.write_text('\n'.join(rows)+'\n',encoding='utf-8')
    def forbidden(_):
        pytest.fail('truth accessed before complete predictions')
    with pytest.raises((ValueError,FileNotFoundError)):
        score_fixed_predictions(directory,forbidden)


def test_batch_caps_and_duplicate_input_ids_fail_before_model(tmp_path):
    model = IQClassifier()
    records = fixture_records()
    limited = args(); limited.eval_max_batches=1
    with pytest.raises(ValueError,match='complete'):
        write_fixed_predictions(model,records,{'all':[r.record_id for r in records]},limited,
                                torch.device('cpu'),tmp_path/'partial')
    with pytest.raises(ValueError,match='unique'):
        write_fixed_predictions(model,[records[0],records[0]],{'all':['opaque_0']},args(),
                                torch.device('cpu'),tmp_path/'duplicate')
    assert model.calls == 0


def test_actual_satellite_transform_path_writes_every_scene(tmp_path):
    records = fixture_records()
    directory = tmp_path/'actual_sat'
    write_fixed_predictions(IQClassifier(),records,{'all':[r.record_id for r in records]},args(),
                            torch.device('cpu'),directory)
    report = score_fixed_predictions(directory,lambda ids:{rid:i%2 for i,rid in enumerate(ids)})
    assert report['status'] == 'COMPLETE' and report['prediction_rows'] == 16
    clean = (directory/'predictions.clean.jsonl').read_text()
    rain = (directory/'predictions.leo_rain_weak.jsonl').read_text()
    assert [json.loads(x)['logits'] for x in clean.splitlines()] != [json.loads(x)['logits'] for x in rain.splitlines()]


def test_wisig_received_adapter_matches_model_input_and_deduplicates_aliases(tmp_path,monkeypatch):
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    from scripts.verify_core90_cross_response import make_fixture
    path = tmp_path/'physical.pkl'
    make_fixture(path,records_per_cell=2)
    ds = WiSigCompactDataset(load_wisig_compact_pkl(str(path)),rx_keep=[4],day_keep=[1],equalized=1)
    expected = [ds[i][0] for i in range(len(ds))]
    def no_label_api(*_):
        pytest.fail('label-returning dataset API used during frozen prediction')
    monkeypatch.setattr(WiSigCompactDataset,'__getitem__',no_label_api)
    loaders = {'test_unseen_day_unseen_rx':SimpleNamespace(dataset=ds),
               'test_unseen_day_rx_4':SimpleNamespace(dataset=ds)}
    records,named,truth = wisig_received_catalog(loaders)
    assert len(records) == len(ds)
    assert named['test_unseen_day_unseen_rx'] == named['test_unseen_day_rx_4']
    for i,record in enumerate(records):
        torch.testing.assert_close(record.read_iq(),expected[i],rtol=0,atol=0)
    assert len(truth([r.record_id for r in records])) == len(ds)
    ds.index.append(ds.index[0])
    with pytest.raises(ValueError,match='duplicate physical'):
        wisig_received_catalog(loaders)


def deployment_checkpoint_parity(checkpoint_path,device='cpu'):
    """Strictly load the real saved identity model without auxiliary runtime state."""
    from SSDG import train_ssdg as train
    from copy import deepcopy
    payload = torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    saved = SimpleNamespace(**payload['args'])
    model_args = train.merge_checkpoint_args(payload,saved,input_len=256,num_domains=4)
    model_args = train._apply_model_cli_args(model_args,saved)
    model = train.build_baseline_model(model_args,torch.device(device))
    model.load_state_dict(payload['model'],strict=True)
    clone = train.build_baseline_model(model_args,torch.device(device))
    clone.load_state_dict(deepcopy(model.state_dict()),strict=True)
    model.eval(); clone.eval()
    torch.manual_seed(744)
    x = torch.randn(3,2,256,device=device)
    with torch.no_grad():
        rich = model(x,y_tx=None,domain_labels=None,return_aux=True)
        deployed = clone(x,y_tx=None,domain_labels=None,return_aux=False)
    torch.testing.assert_close(deployed,rich['tx_logits'],rtol=0,atol=0)
    assert not any('cross_response' in key or 'auxiliary' in key for key in clone.state_dict())
    return {'status':'STRICT_DEPLOYMENT_PARITY','classes':deployed.shape[1],
            'samples':deployed.shape[0],'auxiliary_runtime_loaded':False}


@pytest.mark.skipif(not os.environ.get('CORE90_SYNTHETIC_CHECKPOINT'),reason='requires generated real synthetic checkpoint')
def test_real_checkpoint_strict_deployment_output_parity():
    result = deployment_checkpoint_parity(os.environ['CORE90_SYNTHETIC_CHECKPOINT'])
    assert result['status'] == 'STRICT_DEPLOYMENT_PARITY'
