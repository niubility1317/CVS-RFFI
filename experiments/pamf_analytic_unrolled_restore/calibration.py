import numpy as np
import torch
from . import design as d
from .descriptors import fit_scales,validate_scales
from experiments.pamf_restoration.runner import read_json,array_tensor
from experiments.pamf_restoration.data import write_json


def validate_metadata(value):
    if not isinstance(value,dict):raise ValueError('Source-training scale metadata required')
    expected=dict(schema='training_clean_analytic_scale_v1',status='SOURCE_TRAINING_SCALES_FIXED',role='train',count=8192,
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,V_reads=0,target_reads=0,optimizer_updates=0,
        data_contract='original7/63/30',order='original_first8192',floor=.05)
    if any(value.get(key)!=item for key,item in expected.items()):raise ValueError('Fixed original source clean scale permissions/count differ')
    labelled,unlabelled=value.get('L_count',-1),value.get('U_count',-1)
    if not 0<labelled<=6300 or not 0<unlabelled<=56700 or labelled+unlabelled!=8192:
        raise ValueError('Original L/U-only scale exposure required')
    scales={key:torch.tensor(values,dtype=torch.float64) for key,values in value.get('scales',{}).items()}
    validate_scales(scales);return scales


def calibrate(args):
    if args.parent_action.name!=d.ACTION:raise ValueError('Original R07 source cache required')
    cache=args.parent_action/'cache';meta=read_json(cache/'complete.json')
    if meta['status']!='ACTION_SOURCE_CACHE_COMPLETE' or meta['train_count']!=63000 or meta['TX_labels_read'] or meta['target_reads']:
        raise ValueError('Existing complete source-only cache required')
    clean=np.load(cache/'train/clean.npy',mmap_mode='r',allow_pickle=False);roles=np.load(cache/'train/roles.npy',allow_pickle=False)
    if clean.shape!=(63000,2,256) or roles.shape!=(63000,) or (roles=='L_s').sum()!=6300 or (roles=='U_s').sum()!=56700:
        raise ValueError('Full original L/U roles without V required')
    scales=fit_scales(array_tensor(clean[start:min(start+128,8192)],'cpu') for start in range(0,8192,128))
    value=dict(schema='training_clean_analytic_scale_v1',status='SOURCE_TRAINING_SCALES_FIXED',role='train',count=8192,
        L_count=int((roles[:8192]=='L_s').sum()),U_count=int((roles[:8192]=='U_s').sum()),
        TX_labels_read=0,F_outputs_read=0,F_weights_read=0,V_reads=0,target_reads=0,optimizer_updates=0,
        data_contract='original7/63/30',order='original_first8192',floor=.05,
        scales={key:values.tolist() for key,values in scales.items()})
    validate_metadata(value);write_json(args.output/'analytic_scale.json',value);return value
