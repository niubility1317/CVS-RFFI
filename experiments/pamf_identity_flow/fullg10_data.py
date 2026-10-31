"""Label-blind loading contract and paired scheduling for full source G10."""
import numpy as np


EXPECTED=dict(schema='p3_restore_source_cache_v1',status='VALIDATED_ONCE',
    capsule_id='p3restore-LU392005-G10MIX01-clean-pairs-20261101-r01',split_id='392005-LsUs-source5rx-3day-eq1',
    run_id='20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01',code_commit='53410ce889adc94c53c1783741f970fbdaf51453',
    count=63000,counts={'L_s':6300,'U_s':56700},shape=[63000,2,256],dtype='float32',order='originalL_s_then_originalU_s',
    episode={'id':'MIX01','scene_seed':2026103001,'channel_seed':2026104001},received_path='originalG10P0',
    clean_path='rawFP64_to_FP32_to_oneRMS',TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0)


def validate_manifest(manifest):
    if not isinstance(manifest,dict) or any(manifest.get(k)!=v for k,v in EXPECTED.items()):
        raise ValueError('Exact completed label-blind G10 source pair manifest required')


def mix_full_g10(clean,received,order):
    n=len(order)
    if n%4:raise ValueError('Fixed 75% G10 and 25% clean source schedule required')
    split=3*n//4
    x=np.array(clean[order],dtype=np.float32,copy=True)
    y=np.array(received[order],dtype=np.float32,copy=True)
    y[split:]=x[split:]
    return x,y,dict(g10=split,clean=n-split)
