"""A frozen source-selected restorer, independent target waveform scoring.

This is explicitly the capsule's permitted Phase1 paired diagnostic. A
builder creates the six declared practical observations. The predictor sees
only the named observation for each view and never waveform error or TX truth.
No output here is a p2_min_v1 capsule or evidence of Phase2 protocol compliance.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from . import design as d
from .data import SCENES, write_json


def target_truth_path(capsule):
    """Resolve the two established dataset layouts after predictions finish.

    Inspect paths only; the independent scorer is the first truth reader.
    Never silently choose when both layouts supply a truth file.
    """
    capsule = Path(capsule)
    candidates = (capsule.parent/'truth.json', capsule.parent.parent/'scoring/truth.json')
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError('Exactly one declared target truth layout must exist')
    return present[0]


def confirm(name, capsule, output, frozen_bundle, source_cache):
    from .runner import ROOT, read_json, predict_row
    from leo_practical.channel import Config, normalize_input
    from leo_practical.batch import apply_leo_practical_channel_batch
    decision = read_json(output/'source_decision.json')
    if decision['status'] != 'SOURCE_SELECTION_FROZEN' or decision['selected'] != name:
        raise ValueError('Source selection must be frozen before target access')
    capsule = Path(capsule)
    manifest = read_json(capsule/'manifest.json')
    if (manifest['status'] != 'VALIDATED_ONCE' or manifest['count'] != 168000 or
            not manifest['source_target_disjoint'] or not manifest['unique_ids'] or
            manifest['channel'] != 'residual/post_sync/noeq' or
            manifest['clean_role'] != 'Phase1 paired diagnostic only; never exported to Phase2'):
        raise ValueError('Matching validated Phase1 target capsule required')
    with np.load(capsule/'index.npz', allow_pickle=False) as archive:
        all_ids = archive['ids']
    # Fixed uniformly drawn physical panel. No class/receiver/performance
    # stratification or truth-side selection is used.
    chosen = np.sort(np.random.default_rng(d.TARGET_PANEL_SEED).permutation(len(all_ids))[:d.TARGET_PANEL_SIZE])
    ids = all_ids[chosen]
    raw = np.load(capsule/'clean.npy', mmap_mode='r', allow_pickle=False)
    if raw.shape != (len(all_ids), 2, 256):
        raise ValueError('Target shape differs')
    clean = np.stack([np.stack((z.real, z.imag)) for z in
                      (normalize_input(v[0]+1j*v[1])[0] for v in raw[chosen])]).astype(np.float32)
    reference = output/'target_inputs'
    reference.mkdir()
    np.save(reference/'clean.npy', clean, allow_pickle=False)
    np.save(reference/'physical_ids.npy', ids, allow_pickle=False)
    # The opaque capsule contains no physical RX/day metadata. Do not read
    # scorer truth or reverse opaque IDs to manufacture stratification.
    np.save(reference/'rx.npy', np.full(len(ids), -1, np.int64), allow_pickle=False)
    np.save(reference/'day.npy', np.full(len(ids), -1, np.int64), allow_pickle=False)
    for scene in SCENES:
        cfg = Config(fs_hz=25e6, fc_hz=2462e6, scenario=scene, processing_route='residual',
                     mode='post_sync', equalization_enabled=False, reference_snr_db=30., boundary='reflect')
        # No target RX sessions are exposed in this capsule. Use an explicitly
        # fixed common receiver session for this synthetic paired diagnostic;
        # do not represent it as the historical satellite observation.
        received, _, _ = apply_leo_practical_channel_batch(
            clean, cfg, seed=d.EVALUATION_SEED, sample_ids=ids.tolist(),
            session_ids=['restoration-confirmation-fixed-receiver']*len(ids),
            realization_namespace='restoration/target-confirmation/draw0',
            receiver_seed=d.AUGMENTATION_SEED, normalize_input_rms=False, return_meta=False)
        np.save(reference/(scene+'.npy'), received, allow_pickle=False)
        print(json.dumps(dict(event='TARGET_VIEW_BUILT', scene=scene, count=len(ids))), flush=True)
    write_json(reference/'manifest.json', dict(status='PHASE1_PAIRED_DIAGNOSTIC_COMPLETE', count=len(ids),
               source_selected_candidate=name, capsule=str(capsule), target_panel_seed=d.TARGET_PANEL_SEED,
               scenes=list(SCENES), waveform_references_for_scorer=True, TX_truth_reads=0,
               RX_day_metadata='unavailable; -1 denotes unknown, not a physical receiver',
               added_receiver_session='fixed common synthetic receiver; no historical session claim',
               not_Phase2=True))
    folder = output/'target_confirmation'/name
    folder.mkdir(parents=True)
    shutil.copy2(output/name/'final.pt', folder/'final.pt')
    predict_row(name, reference, folder, 'cuda:0', frozen_bundle, source_cache)
    subprocess.run([sys.executable, '-X', 'utf8', '-m', 'experiments.pamf_restoration.runner',
                    'score', '--reference', str(reference), '--row-output', str(folder),
                    '--target-truth', str(target_truth_path(capsule))],
                   cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    write_json(output/'target_confirmation/complete.json', dict(status='ANALYZED_PHASE1_PAIRED_DIAGNOSTIC',
               candidate=name, count=len(ids), views=['clean', *SCENES],
               selection_before_target=True, scored_target_TX_labels=len(ids), identity_accuracy_measured=True,
               no_target_feedback_to_training_or_selection=True))
