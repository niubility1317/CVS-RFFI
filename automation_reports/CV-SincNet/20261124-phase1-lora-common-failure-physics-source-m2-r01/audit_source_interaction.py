"""Posthoc paired estimate of the noise removal by inverse interaction."""
import json
from pathlib import Path

import numpy as np

from experiments.cric_dura_comparison.io import read
from experiments.pamf_lora_main.chirp_spectrum_source import read_frozen
from experiments.pamf_lora_main.common_failure_source import MODES, ROUTES
from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.full_packet_views import source_contract


def main():
    runtime = read(Path('runtime.json'))
    root, source = Path(runtime['run_root']), Path(runtime['source_root'])
    if read(root/'state.json')['status'] != 'SOURCE_SCORED':
        raise ValueError('Source results are not terminal')
    frozen = {r: read_frozen(root, r, MODES) for r in ROUTES}
    ids = frozen['full'][0]
    if ids != frozen['residual'][0]:
        raise ValueError('Paired source routes have different packet IDs')
    positions = {key: i for i, key in enumerate(source_contract(source)['role_ids']['V'])}
    chosen = [positions[key] for key in ids]
    with np.load(source/'V_metadata.npz', allow_pickle=False) as metadata:
        if np.asarray(ids).tolist() != metadata['ids'][chosen].tolist():
            raise ValueError('Source metadata and prediction IDs do not agree')
        labels = metadata['y'][chosen]
    per_rx = len(ids)//len(SOURCE)
    rng = np.random.default_rng(2026113641)
    sample = np.concatenate([rng.integers(i*per_rx,(i+1)*per_rx,(4000,per_rx)) for i in range(len(SOURCE))],axis=1)
    outcomes = {}
    for route, (_, rows, _) in frozen.items():
        hits = np.stack([rows[k] == labels for k in VIEWS[1:]],axis=1).astype(np.int32)
        raw, corrected = (hits[MODES.index(k)] for k in ('direct','action'))
        base_difference = corrected - raw
        effect = {}
        for dose in ('noise80','noise100'):
            no_inverse, inverse = (hits[MODES.index(k)] for k in (dose,dose+'_action'))
            interaction = (inverse - no_inverse) - base_difference
            results = {}
            for name, vector in (('six',interaction.sum(axis=0)),('low',interaction[-1])):
                results[name] = dict(net=int(vector.sum()),
                                     interval=np.quantile(vector[sample].sum(axis=1),(.025,.975)).tolist())
            effect[dose] = results
        clean = rows['clean']
        outcomes[route] = dict(interaction=effect, clean_diagnostic_disagreements=int(np.sum(clean[2:]!=clean[:1])))
    print(json.dumps(dict(status='VERIFIED',status_scope='completed source scores',routes=outcomes,
                          paired_bootstrap_draws=4000,target_reads=0)))


if __name__ == '__main__':
    main()
