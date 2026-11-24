"""Collect a frozen LoRa Tri screen and score the fixed target split independently."""
import argparse
from pathlib import Path
import shlex
from statistics import mean
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import receive, wait_for_predictions
from .data import CLASSES, TARGET, VIEWS, label_ids, packet_indices, recording
from .full_packet_data import LIMIT
from .scoring import macro_f1
from .tri_tuning import ROWS, ROWS2, SEED


def load_predictions(root, names):
    if not names or names[0] != 'control' or len(names) != len(set(names)) or any(n not in ROWS and n not in ROWS2 for n in names):
        raise ValueError('Invalid frozen LoRa selection')
    results = {}
    for name in names:
        folder = root/name
        if read(folder/'complete.json') != dict(method=name, seed=SEED, status='PREDICTIONS_FROZEN',
                target_truth_read=False, receivers=list(TARGET), per_receiver=LIMIT, views=list(VIEWS)):
            raise ValueError(f'Incomplete prediction declaration {name}')
        for view in VIEWS:
            shards = []
            for rx in TARGET:
                pred = np.load(folder/f'{rx}__{view}__direct.npy', allow_pickle=False)
                if pred.shape != (LIMIT,) or pred.dtype.kind not in 'iu' or (pred < 0).any() or (pred >= len(CLASSES)).any():
                    raise ValueError(f'Invalid prediction vector {name}/{rx}/{view}')
                shards.append(pred)
            results[name, view] = np.concatenate(shards)
    return results


def score(predict, names, truth):
    result = {}
    for name in names:
        metrics = {}
        for view in VIEWS:
            pred = predict[name, view]
            ok, old = pred == truth, predict['control', view] == truth
            metrics[view] = dict(n=len(truth), correct=int(ok.sum()), accuracy=float(ok.mean()),
                                 macro_f1=macro_f1(truth, pred), repairs=int((ok & ~old).sum()),
                                 harms=int((old & ~ok).sum()),
                                 by_rx=dict(zip(TARGET,ok.reshape(len(TARGET),LIMIT).mean(1).tolist(),strict=True)),
                                 by_tx={str(tx):float(ok[truth==i].mean()) for i,tx in enumerate(CLASSES)})
        result[name] = dict(views=metrics, s6=mean(metrics[v]['accuracy'] for v in VIEWS[1:]))
    return dict(status='SCORED_COMPLETE', seed=SEED, truth_last=True, results=result)


def collect(submit, ssh_config, dataset, output):
    remote = shlex.quote(read(submit)['run_root'])
    conn = ['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    state = wait_for_predictions(conn, remote)
    output.mkdir(parents=True, exist_ok=False)
    receive(conn, remote, output, main=False)
    names = state['names']
    predictions = load_predictions(output/'predictions', names)
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        truth = []
        for i, rx in enumerate(TARGET):
            with recording(archive,f'{rx}_test.h5',output) as h5:
                truth.append(label_ids(h5,LIMIT)[packet_indices(LIMIT,i)])
    write(output/'scores.json',score(predictions, names, np.concatenate(truth)))
    print('LORA_TRI_SCORED', output/'scores.json',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('submit','ssh-config','dataset','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    collect(**vars(parser.parse_args()))
