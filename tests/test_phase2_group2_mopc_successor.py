import json

import pytest

from experiments.pamf_phase2_practical import group2_mopc_successor as successor


def write(path, data):
    path.write_text(json.dumps(data), encoding='utf-8')


def test_completed_and_quarantined_never_enter_new_queue(tmp_path):
    partition = dict(completed=[dict(key='done', output='old/done')],
                     quarantined=[dict(key='bad', output='old/bad')],
                     pending=[dict(key='new', output='new/new', profile='dadda_sda', methods=['mopc_hr'])])
    parent = dict(reused=[dict(key='old')], deferred=[dict(key='older-bad')],
                  cancelled=[dict(key='radio')])
    manifest, queue = successor.successor_manifest(parent, partition, '/old')
    assert [job['key'] for job in queue] == ['new']
    assert [job['key'] for job in manifest['completed_mopc']] == ['done']
    assert [job['key'] for job in manifest['quarantined_mopc']] == ['bad']
    assert manifest['deferred'] == parent['deferred'] and manifest['cancelled'] == parent['cancelled']
    assert manifest['truth_read'] is False and manifest['query_fit'] is False


def test_successor_rejects_category_overlap(tmp_path):
    parent = dict(reused=[], deferred=[], cancelled=[])
    partition = dict(completed=[dict(key='same')], quarantined=[], pending=[dict(key='same')])
    with pytest.raises(ValueError, match='overlap'):
        successor.successor_manifest(parent, partition, '/old')


def test_successor_rejects_radio_or_wrong_methods():
    parent = dict(reused=[], deferred=[], cancelled=[])
    for job in (dict(key='radio', profile='radionet_ada_knn', methods=['mopc_hr']),
                dict(key='x', profile='dadda_sda', methods=['orthogonal'])):
        with pytest.raises(ValueError, match='pending'):
            successor.successor_manifest(parent, dict(completed=[], quarantined=[], pending=[job]), '/old')
