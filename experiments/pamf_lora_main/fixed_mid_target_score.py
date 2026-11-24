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


def read_matrix(output, controls, modes=(MODE,)):
    expected = {f'{rx}__{view}__{mode}.npy' for rx in TARGET for view in VIEWS for mode in modes}
    matrices, activity = {}, {}
    for route, control in zip(ROUTES, controls, strict=True):
        path = output/'predictions'/route
        if (read(path/'complete.json') != freeze_metadata(route, modes)
                or {entry.name for entry in path.glob('*.npy')} != expected):
            raise ValueError(f'Incomplete frozen {route} prediction matrix')
        control_metadata(control, route)
        activity[route] = read(path/'activity.json')
        if (set(activity[route]) != set(TARGET)
                or any(set(views) != set(VIEWS) or any(type(n) is not int or not 0 <= n <= LIMIT
                    for n in views.values()) for views in activity[route].values())):
            raise ValueError(f'Incomplete {route} gate activity')
        matrices[route] = {}
        for mode in (*CONTROL_MODES[route], *modes):
            parent = path if mode in modes else control
            matrices[route][mode] = np.asarray([
                [frozen_prediction(output_path(parent, rx, view, mode)) for view in VIEWS] for rx in TARGET])
        direct = matrices[route]['direct']
        for mode in modes:
            candidate = matrices[route][mode]
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


def compare_hits(candidate, baseline, draws):
    six = paired_counts(candidate[:,1:], baseline[:,1:])
    six['ci95'] = interval(candidate[:,1:].sum(1), baseline[:,1:].sum(1), draws)
    views = {}
    for i, view in enumerate(VIEWS):
        result = paired_counts(candidate[:,i], baseline[:,i])
        result['ci95'] = interval(candidate[:,i], baseline[:,i], draws)
        views[view] = result
    return dict(six=six, views=views,
                receivers={rx: {view: paired_counts(candidate[r,i],baseline[r,i])
                                for i,view in enumerate(VIEWS)} for r,rx in enumerate(TARGET)})


def summarize(rows, truth, draws, *, primary=MODE, additional_pairs=None):
    hits = {name: pred == truth[:,None,:] for name,pred in rows.items()}
    raw = hits['direct']
    values = {}
    for mode, found in hits.items():
        values[mode] = dict(
            clean=paired_counts(found[:,0],raw[:,0]), six=paired_counts(found[:,1:],raw[:,1:]),
            views={view:paired_counts(found[:,i],raw[:,i]) for i,view in enumerate(VIEWS)},
            receivers={rx:{view:paired_counts(found[r,i],raw[r,i]) for i,view in enumerate(VIEWS)}
                       for r,rx in enumerate(TARGET)})
    proposed = hits[primary]
    paired = {name: compare_hits(proposed, baseline, draws)
              for name,baseline in hits.items() if name != primary}
    result = dict(modes=values, candidate_vs=paired)
    if additional_pairs:
        result['dose_matched_pairs'] = {
            candidate: dict(reference=baseline, **compare_hits(hits[candidate], hits[baseline], draws))
            for candidate,baseline in additional_pairs.items()
        }
    return result


def add_previous(matrices, activity, previous, previous_activity, names):
    for route in ROUTES:
        current, historical = matrices[route], previous[route]
        if (activity[route] != previous_activity[route]
                or any(not np.array_equal(current[k], historical[k]) for k in CONTROL_MODES[route])):
            raise ValueError(f'Historical width controls differ in prediction alignment or activity for {route}')
        if set(names) & current.keys():
            raise ValueError(f'Duplicate historical width mode for {route}')
        current.update((name, historical[name]) for name in names)


def collect(submit, ssh_config, dataset, output, full_control, residual_control,
            *, run=RUN, modes=(MODE,), primary=MODE, previous=None, additional_pairs=None):
    receipt = read(submit)
    remote = PurePosixPath(receipt['run_root'])
    if remote.name != run or receipt['archive_transfer'] != 'VERIFIED' or receipt['remote_compile'] != 'PASS':
        raise ValueError('Wrong official LoRa target release')
    connection = ['ssh','-F',str(ssh_config),'-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607']
    wait_for_predictions(connection, shlex.quote(str(remote)))
    output.mkdir(parents=True)
    receive(connection, shlex.quote(str(remote)), output, main=False)
    controls = (full_control, residual_control)
    matrices, activity = read_matrix(output, controls, modes)
    if previous is not None:
        historical_path, historical_modes = previous
        historical, historical_activity = read_matrix(historical_path, controls, historical_modes)
        names = tuple(dict.fromkeys(additional_pairs.values()))
        if not set(additional_pairs).issubset(modes) or not set(names).issubset(historical_modes):
            raise ValueError('Dose-matched modes differ from the frozen target matrix')
        add_previous(matrices, activity, historical, historical_activity, names)
    truth = []
    with zipfile.ZipFile(dataset/'multiple_receiver_test.zip') as archive:
        for i, rx in enumerate(TARGET):
            with recording(archive, f'{rx}_test.h5', output) as data:
                truth.append(label_ids(data, LIMIT)[packet_indices(LIMIT, i)])
    y = np.stack(truth)
    draws = np.random.default_rng(392240).integers(LIMIT,size=(len(TARGET),1600,LIMIT),dtype=np.int16)
    scores = {r:summarize(matrix,y,draws,primary=primary,additional_pairs=additional_pairs)
              for r,matrix in matrices.items()}
    clean_switch = int(np.count_nonzero(matrices['full'][primary][:,0] != matrices['residual'][primary][:,0]))
    write(output/'scores.json',dict(status='SCORED_COMPLETE', seed=SEED, n_packets=y.size,
          truth_after_complete_dual_predictions=True, clean_route_label_disagreements=clean_switch,
          activity=activity, routes=scores))
    print('LORA_TARGET_SCORED', output/'scores.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('submit','ssh-config','dataset','output','full-control','residual-control'):
        parser.add_argument('--'+key,type=Path,required=True)
    collect(**vars(parser.parse_args()))
