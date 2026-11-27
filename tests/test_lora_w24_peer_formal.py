import json

import numpy as np
import pytest

from experiments.pamf_lora_main import w24_formal as formal
from experiments.pamf_lora_main import w24_peer_spectral_direction_source as source


def test_peer_formal_keeps_locked_full_packet_matrix():
    expected = list(formal.MODES[formal.FULL][:-1]) + [formal.WIDE_UNGATED, *source.MODES[2:]]
    matrix = formal.cases(formal.PEER_SPECTRAL_KIND)
    assert list(matrix.values()) == [formal.FULL] * 3
    rows = [formal.receipt(case, True, third_prior=formal.PEER_SPECTRAL_KIND) for case in matrix]
    assert [rx for row in rows for rx in row['receivers']] == list(formal.TARGET)
    assert all(row['modes'] == expected and row['seeds'] == [2026092702]
               and row['per_receiver'] == 2000 and row['target_truth_read'] is False for row in rows)


def peer_runtime(project):
    runs = project/'runs'
    root = runs/source.RUN
    root.mkdir(parents=True)
    return root, dict(project=str(project), run_id=formal.RUN_PEER_SPECTRAL_CPU,
                       run_root=str(runs/formal.RUN_PEER_SPECTRAL_CPU),
                       candidate_root=str(root), third_prior=formal.PEER_SPECTRAL_KIND)


def write(path, obj):
    path.write_text(json.dumps(obj), encoding='utf-8')


def test_peer_target_guard_waits_or_declines_source_without_opening_target(tmp_path):
    root, rt = peer_runtime(tmp_path)
    write(root/'state.json', {'status': 'RUNNING', 'phase': 'fit-source'})
    assert formal.peer_source_ready(rt) is None
    write(root/'state.json', {'status': 'SOURCE_SCORED', 'target_reads': 0, 'promote': False})
    assert formal.peer_source_ready(rt) is False
    with pytest.raises(ValueError, match='source qualification'):
        formal.verify_matched_source(rt)
    write(root/'state.json', {'status': 'FAILED', 'error': 'fit-source failed'})
    with pytest.raises(ValueError, match='source run failed'):
        formal.peer_source_ready(rt)


def test_peer_positive_marker_is_rechecked_against_source_truth_last_gate(tmp_path):
    root, rt = peer_runtime(tmp_path)
    write(root/'state.json', {'status': 'SOURCE_SCORED', 'target_reads': 0, 'promote': True})
    waveform = {'variant': source.RUN, 'base': source.PRACTICAL_RUN,
                'received_only': True, 'matched_center_edge_energy': True}
    write(root/'source-predictions.json', dict(status='PREDICTIONS_FROZEN', physical_ids=list(range(300)),
          views=list(source.VIEWS), modes=list(source.MODES), target_reads=0, source_truth_reads=0,
          waveform=waveform))
    score = dict(status='ANALYZED', n_packets=300, n_practical=1800, source_truth_last=False,
                 target_reads=0, waveform=waveform, scores={
                     'direct': {'by_scene': {'clean': 297}},
                     'fixed_dose_direction': {'six_correct': 1450, 'low_urban_correct': 168},
                     'within_chirp': {'six_correct': 1455, 'low_urban_correct': 171},
                     'cross_chirp': {'six_correct': 1470, 'low_urban_correct': 176, 'by_scene': {'clean': 297}},
                 }, paired_contrasts={
                     'cross_chirp_vs_fixed_dose_direction': {'CI95_net_six': [5, 35]},
                     'cross_chirp_vs_within_chirp': {'net_six': 15},
                 })
    write(root/'source-scores.json', score)
    with pytest.raises(ValueError, match='source qualification'):
        formal.verify_matched_source(rt)
    score['source_truth_last'] = True
    write(root/'source-scores.json', score)
    assert formal.peer_source_ready(rt) is True
    formal.verify_matched_source(rt)
    score['scores']['cross_chirp']['low_urban_correct'] = 169
    write(root/'source-scores.json', score)
    with pytest.raises(ValueError, match='source qualification'):
        formal.verify_matched_source(rt)


def test_peer_formal_reuses_frozen_source_forward_and_orders_matched_arms(monkeypatch):
    monkeypatch.setattr(source, 'receive_readouts', lambda *args: list(range(5)))
    monkeypatch.setattr(formal, 'classify_aperture', lambda head, value: np.full(3, value, np.int16))
    decisions, activity = formal.peer_spectral_labels([object()], (None, None, None), np.empty((3, 8192)))
    assert activity == 3 and decisions.dtype == np.int16
    assert decisions.tolist() == [[[0]*3, [2]*3, [3]*3, [4]*3]]


def test_peer_dispatch_declines_before_accessing_target_reference(tmp_path, monkeypatch):
    root, rt = peer_runtime(tmp_path)
    write(root/'state.json', {'status': 'SOURCE_SCORED', 'target_reads': 0, 'promote': False})
    archive = tmp_path/'e104-release'
    archive.mkdir()
    write(archive/'submit.json', {'code_commit': formal.W24_COMMIT})
    runtime = tmp_path/'runtime.json'
    write(runtime, dict(rt, ungated=True, w24_release_root=str(archive)))

    def forbidden(*args):
        raise AssertionError('Unqualified direction must not touch the target reference')

    monkeypatch.setattr(formal, 'blind_reference', forbidden)
    formal.dispatch(runtime)
    out = tmp_path/'runs'/formal.RUN_PEER_SPECTRAL_CPU
    assert json.loads((out/'state.json').read_text()) == dict(status='SOURCE_NOT_QUALIFIED', target_truth_read=False)
    assert list((out/'predictions').iterdir()) == []


def test_peer_collector_does_not_download_or_score_unqualified_source(tmp_path, monkeypatch):
    import subprocess
    from experiments.pamf_lora_main import w24_peer_formal_collect as collector

    submit = tmp_path/'submit.json'
    write(submit, {'run_root': f'/tmp/runs/{formal.RUN_PEER_SPECTRAL_CPU}'})
    output = tmp_path/'local'
    calls = []

    def status_only(argv, **kwargs):
        calls.append(argv[0])
        assert argv[0] == 'ssh'
        return subprocess.CompletedProcess(argv, 0, b'{"status":"SOURCE_NOT_QUALIFIED","target_truth_read":false}')

    monkeypatch.setattr(collector.subprocess, 'run', status_only)
    collector.collect(submit, tmp_path/'absent-dataset', output, tmp_path/'ssh-config')
    assert calls == ['ssh'] and not output.exists()


def test_full_handoff_routes_negative_source_to_same_matrix(tmp_path):
    root, rt = peer_runtime(tmp_path)
    previous = tmp_path/'runs'/formal.RUN_PEER_SPECTRAL_CPU
    (previous/'predictions').mkdir(parents=True)
    run = formal.RUN_PEER_SPECTRAL_FULL
    rt.update(run_id=run, run_root=str(previous.parent/run))
    assert formal.cpu_formal(rt)
    write(previous/'state.json', {'status': 'WAITING_FOR_PEER_SOURCE_SCORE'})
    assert formal.peer_formal_handoff(rt) is None
    write(previous/'state.json', {'status': 'SOURCE_NOT_QUALIFIED', 'target_truth_read': False})
    write(root/'state.json', {'status': 'SOURCE_SCORED', 'promote': False, 'target_reads': 0})
    assert formal.peer_formal_handoff(rt) == 'PREDICT'
    formal.verify_matched_source(rt)
    (previous/'predictions'/'unexpected').mkdir()
    with pytest.raises(ValueError, match='Inconsistent'):
        formal.verify_matched_source(rt)


def test_full_handoff_reuses_frozen_legacy_rows_without_opening_target(tmp_path, monkeypatch):
    _, rt = peer_runtime(tmp_path)
    previous = tmp_path/'runs'/formal.RUN_PEER_SPECTRAL_CPU
    cases = formal.cases(formal.PEER_SPECTRAL_KIND)
    for case in cases:
        shard = previous/'predictions'/case
        shard.mkdir(parents=True)
        write(shard/'complete.json', formal.receipt(case, True, third_prior=formal.PEER_SPECTRAL_KIND))
    write(previous/'state.json', {'status': 'PREDICTIONS_FROZEN', 'cases': list(cases), 'target_truth_read': False})
    run = formal.RUN_PEER_SPECTRAL_FULL
    archive = tmp_path/'e104-release'
    archive.mkdir()
    write(archive/'submit.json', {'code_commit': formal.W24_COMMIT})
    runtime = tmp_path/'runtime.json'
    out = previous.parent/run
    write(runtime, dict(rt, run_id=run, run_root=str(out), ungated=True, w24_release_root=str(archive)))

    def forbidden(*args):
        raise AssertionError('The handoff must preserve prior ownership when the same predictions exist')

    monkeypatch.setattr(formal, 'blind_reference', forbidden)
    formal.dispatch(runtime)
    assert json.loads((out/'state.json').read_text()) == dict(status='MATCHED_FULL_EXISTS',
        previous_run=formal.RUN_PEER_SPECTRAL_CPU, target_truth_read=False)
    assert list((out/'predictions').iterdir()) == []


def test_peer_score_requires_frozen_shards_before_truth(tmp_path, monkeypatch):
    from experiments.pamf_lora_main import w24_formal_score as scorer

    def forbidden(*args):
        raise AssertionError('Truth must remain unopened before all shards close')

    monkeypatch.setattr(scorer.zipfile, 'ZipFile', forbidden)
    with pytest.raises(FileNotFoundError):
        scorer.score(tmp_path/'incomplete', tmp_path/'truth', tmp_path/'score',
                     third_prior=formal.PEER_SPECTRAL_KIND)
