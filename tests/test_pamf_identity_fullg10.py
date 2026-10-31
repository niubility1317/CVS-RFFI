import numpy as np
import pytest
import torch
from experiments.pamf_identity_flow.fullg10_data import validate_manifest, mix_full_g10
from experiments.pamf_identity_flow import fullg10_rollout_train


def test_clean_and_g10_mixture_preserves_paired_order():
    clean=np.arange(8,dtype=np.float32)[:,None,None]*np.ones((8,2,256),np.float32)
    received=clean+5;order=np.array([7,5,3,1,6,4,2,0])
    x,y,counts=mix_full_g10(clean,received,order)
    assert counts=={'g10':6,'clean':2}
    assert np.array_equal(x,clean[order])
    assert np.all(y[:6]-x[:6]==5)
    assert np.array_equal(y[6:],x[6:])


def test_manifest_rejects_target_or_wrong_channel():
    manifest=dict(schema='p3_restore_source_cache_v1',status='VALIDATED_ONCE',
        capsule_id='p3restore-LU392005-G10MIX01-clean-pairs-20261101-r01',split_id='392005-LsUs-source5rx-3day-eq1',
        run_id='20261101-data-P3Restore-LU-G10-pairs-s2026092701-r01',code_commit='53410ce889adc94c53c1783741f970fbdaf51453',
        count=63000,counts={'L_s':6300,'U_s':56700},shape=[63000,2,256],dtype='float32',order='originalL_s_then_originalU_s',
        episode={'id':'MIX01','scene_seed':2026103001,'channel_seed':2026104001},
        received_path='originalG10P0',clean_path='rawFP64_to_FP32_to_oneRMS',TX_labels_read=0,OE_reads=0,V_reads=0,target_inputs=0,optimizer_updates=0)
    validate_manifest(manifest)
    for key in ('target_inputs','V_reads','TX_labels_read'):
        invalid=manifest.copy();invalid[key]=1
        with pytest.raises(ValueError):validate_manifest(invalid)
    invalid=manifest.copy();invalid['received_path']='wrong'
    with pytest.raises(ValueError):validate_manifest(invalid)


def test_fullg10_rollout_budget_and_checkpoint_roundtrip(tmp_path):
    assert fullg10_rollout_train.EPOCHS==30 and fullg10_rollout_train.STEPS==14790
    assert fullg10_rollout_train.config()['source_mix']=={'g10':47250,'clean':15750}
    model=fullg10_rollout_train.make_model();path=tmp_path/'model.pt';commit='a'*40
    fullg10_rollout_train.save(path,model,commit,epoch=0,updates=0)
    loaded=fullg10_rollout_train.load(path,'cpu',commit,epoch=0)
    with torch.no_grad():
        x=torch.randn(2,2,256)
        torch.testing.assert_close(loaded(x),x,rtol=1e-6,atol=1e-6)
