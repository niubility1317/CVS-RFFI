import numpy as np

from experiments.pamf_lora_main import w24_formal_score as scorer


def test_transmitter_statistics_use_physical_counts_with_same_packet_repairs():
    # Two receivers, three packets each. TX 0 occurs four times and TX 1 twice.
    truth = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int16)
    labels = truth[:, None, None, None, :]
    pred = np.broadcast_to(labels, (2, 7, 2, 2, 3)).copy()

    # Low-urban within TX 0: two repairs in seed 17 and one harm in seed 18.
    pred[0, -1, 0, 0, :2] = 1
    pred[1, -1, 1, 1, 2] = 1
    # Clean within TX 1: one harm, with no change to six-practical counts.
    pred[0, 0, 0, 1, 2] = 0
    hit = pred == labels
    by_tx = scorer.transmitter_groups(hit, truth, ('within', 'peer'), (17, 18))
    assert set(by_tx) == {'0', '1'}
    assert by_tx['0']['n_physical'] == 4 and by_tx['1']['n_physical'] == 2
    assert by_tx['0']['readouts']['peer']['pooled']['accuracy_by_scene']['practical_low_urban'] == 7/8
    assert by_tx['1']['readouts']['peer']['17']['accuracy_by_scene']['clean'] == 1/2
    assert by_tx['1']['readouts']['peer']['17']['six_accuracy'] == 1
    low = by_tx['0']['paired']['within']['pooled']['practical_low_urban']
    assert low == {'repairs': 2, 'harms': 1, 'net': 1}
    assert by_tx['1']['paired']['within']['17']['six_scene']['net'] == 0

    for i, mode in enumerate(('within', 'peer')):
        total = hit[:, :, :, i].sum((0, 2, 3)).tolist()
        pooled = [x['readouts'][mode]['pooled']['correct_by_scene'] for x in by_tx.values()]
        assert [sum(x[view] for x in pooled) for view in scorer.VIEWS] == total
