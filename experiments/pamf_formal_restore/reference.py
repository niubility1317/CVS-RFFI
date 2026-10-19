import json
from pathlib import Path
import numpy as np
from experiments.pamf_restoration.data import SCENES,write_json

COUNT=168000
CLASSES=['14-10','14-7','20-15','20-19','6-15','8-20']


def prepare_reference(formal,output,expected_count=COUNT):
    formal,output=Path(formal).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Immutable formal reference exists')
    manifest=json.loads((formal/'manifest.json').read_text(encoding='utf-8'))
    expected=dict(status='VALIDATED_ONCE',count=expected_count,scenes=list(SCENES),classes=CLASSES,
                  channel='residual/post_sync/noeq',truth_read=False,rows_share_observations=True,
                  source_target_disjoint=True)
    if any(manifest.get(key)!=value for key,value in expected.items()):
        raise ValueError('Existing formal observation handles differ')
    with np.load(formal/'index.npz',allow_pickle=False) as index:ids=index['ids'].copy()
    if ids.shape!=(expected_count,):raise ValueError('Opaque record count differs')
    paths={scene:formal/(scene+'.npy') for scene in SCENES}
    paths['clean']=Path(manifest['clean_ref'])
    for path in paths.values():
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        if array.shape!=(expected_count,2,256) or array.dtype!=np.float32:
            raise ValueError('Fixed IQ file format differs')
        del array
    output.mkdir()
    for name,path in paths.items():(output/(name+'.npy')).symlink_to(path.resolve())
    np.save(output/'physical_ids.npy',ids,allow_pickle=False)
    # Neither field enters inference. Independent target scorer replaces both
    # from its existing truth after all seven predictions have been fixed.
    for name in ('rx','day'):
        np.save(output/(name+'.npy'),np.full(expected_count,'unavailable',dtype='U11'),allow_pickle=False)
    write_json(output/'manifest.json',dict(status='FORMAL_REFERENCE_READY',count=expected_count,
        views=['clean',*SCENES],formal_manifest=str(formal/'manifest.json'),
        original_IQ_bytes_reused=True,channel_generation_calls=0,TX_truth_reads=0,
        physical_ids_from='Existing formal index ids only',RX_day_placeholder_not_inference=True))
    return output
