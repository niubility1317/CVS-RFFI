"""Score both complete LoRa midpoint routes against independent local truth."""
import argparse
from pathlib import Path, PurePosixPath
import shlex
import zipfile

import numpy as np

from experiments.cric_dura_comparison.io import read, write
from .collect import receive, wait_for_predictions
from .data import TARGET, VIEWS, label_ids, packet_indices, recording
from .fixed_mid_target import (CONTROL_MODES, LIMIT, MODE, RUN, SEED, control_metadata,
                               freeze_metadata, frozen_prediction, output_path)
from .spectral_unroll import ROUTES


def read_matrix(output, controls):
    expected = {f'{rx}__{view}__{MODE}.npy' for rx in TARGET for view in VIEWS}
    matrices, activity = {}, {}
    for route, control in zip(ROUTES, controls, strict=True):
        path = output/'predictions'/route
        if (read(path/'complete.json') != freeze_metadata(route)
                or {entry.name for entry in path.glob('*.npy')} != expected):
            raise ValueError(f'Incomplete frozen {route} prediction matrix')
        control_metadata(control, route)
        activity[route] = read(path/'activity.json')
        if (set(activity[route]) != set(TARGET)
                or any(set(views) != set(VIEWS) or any(type(n) is not int or not 0 <= n <= LIMIT
                    for n in views.values()) for views in activity[route].values())):
            raise ValueError(f'Incomplete {route} gate activity')
        matrices[route] = {}
        for mode in (*CONTROL_MODES[route], MODE):
            parent = path if mode == MODE else control
            matrices[route][mode] = np.asarray([
                [frozen_prediction(output_path(parent, rx, view, mode)) for view in VIEWS] for rx in TARGET])
        direct = matrices[route]['direct']
        candidate = matrices[route][MODE]
        for i, rx in enumerate(TARGET):
            for j, view in enumerate(VIEWS):
                if np.count_nonzero(candidate[i,j] != direct[i,j]) > activity[route][rx][view]:
                    raise ValueError('More switched predictions than active input packets')
    if not np.array_equal(matrices['full']['direct'][:,0],matrices['residual']['direct'][:,0]):
        raise ValueError('Frozen direct clean predictions disagree across paths')
    return matrices, activity


def paired_counts(candidate, control):
    return dict(n=int(candidate.size), correct=int(candidate.sum()), repair=int((candidate & ~control).sum()),
                damage=int((~candidate & control).sum()), net=int(candidate.sum()-control.sum()),
                accuracy=float(candidate.mean()))


def interval(candidate, baseline, draws):
    difference = candidate.astype(np.int16) - baseline
    counts = np.take_along_axis(difference[:,None,:], draws, axis=-1).sum(axis=(0,2))
    return np.quantile(counts,[.025,.975]).tolist()


def summarize(rows, truth, draws):
    hits = {name: pred == truth[:,None,:] for name,pred in rows.items()}
    raw = hits['direct']
    values = {}
    for mode, found in hits.items():
        values[mode] = dict(
            clean=paired_counts(found[:,0],raw[:,0]), six=paired_counts(found[:,1:],raw[:,1:]),
            views={view:paired_counts(found[:,i],raw[:,i]) for i,view in enumerate(VIEWS)},
            receivers={rx:{view:paired_counts(found[r,i],raw[r,i]) for i,view in enumerate(VIEWS)}
                       for r,rx in enumerate(TARGET)})
    paired = {}
    proposed = hits[MODE]
    for name, reference in hits.items():
        if name == MODE:
            continue
        six = paired_counts(proposed[:,1:], reference[:,1:])
        six['ci95'] = interval(proposed[:,1:].sum(1), reference[:,1:].sum(1), draws)
        views = {}
        for i, view in enumerate(VIEWS):
            result = paired_counts(proposed[:,i],reference[:,i])
            result['ci95'] = interval(proposed[:,i], reference[:,i], draws)
            views[view] = result
        paired[name] = dict(six=six, views=views,
                            receivers={rx: {view: paired_counts(proposed[r,i],reference[r,i])
                                        for i,view in enumerate(VIEWS)} for r,rx in enumerate(TARGET)})
    return dict(modes=values, candidate_vs=paired)


def collect(submit, ssh_config, dataset, output, full_control, residual_control):
    receipt = read(submit)
    remote = PurePosixPath(receipt['run_root'])
    if remote.name != RUN or receipt['archive_transfer'] != 'VERIFIED' or receipt['remote_compile'] != 'PASS':
        raise ValueError('Wrong official midpoint release')
    connection = ['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    wait_for_predictions(connection, shlex.quote(str(remote)))
    output.mkdir(parents=True)
    receive(connection, shlex.quote(str(remote)), output, main=False)
    matrices, activity = read_matrix(output, (full_control, residual_control))
    truth = []
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as data:
                truth.append(label_ids(data, LIMIT)[packet_indices(LIMIT, i)])
    y = np.stack(truth)
    draws = np.random.default_rng(392240).integers(LIMIT,size=(len(TARGET),1600,LIMIT),dtype=np.int16)
    scores = {r:summarize(matrix,y,draws) for r,matrix in matrices.items()}
    clean_switch = int(np.count_nonzero(matrices['full'][MODE][:,0] != matrices['residual'][MODE][:,0]))
    write(output/'scores.json',dict(status='SCORED_COMPLETE', seed=SEED, n_packets=y.size,
          truth_after_complete_dual_predictions=True, clean_route_label_disagreements=clean_switch,
          activity=activity, routes=scores))
    print('LORA_FIXED_MID_TARGET_SCORED', output/'scores.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('submit','ssh-config','dataset','output','full-control','residual-control'):
        parser.add_argument('--'+key,type=Path,required=True)
    collect(**vars(parser.parse_args()))
