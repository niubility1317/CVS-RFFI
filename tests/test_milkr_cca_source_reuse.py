import json
import numpy as np
import pytest
from experiments.pamf_phase2_practical.milkr.paired_cca import PairedCCA,DIMS
from experiments.pamf_phase2_practical.milkr_cca_source import reuse_source_checkpoint


def test_reuse_preserves_source_arrays_and_rejects_target_or_contract_ancestor(tmp_path):
    parent=tmp_path/'parent';parent.mkdir();out=tmp_path/'out';out.mkdir()
    parameters=[(np.zeros(d),np.zeros(d),np.eye(d,16),np.eye(d,16)) for d in DIMS]
    PairedCCA(parameters,16).save(parent/'source_cca.npz')
    common=dict(source_contract='F0contract',source_cache='F0L',DURA={'source':'G_F0'},PAMF_sources={'1':'F_F0'},model_seeds=[1],source_cache_epochs=[91,92],augmentation_cache='source_aug')
    previous=dict(common,kernel_method='paired_cca',run_root=str(parent),run_id='source_parent')
    runtime_file=tmp_path/'runtime.json';runtime_file.write_text(json.dumps(previous))
    selected=dict(status='SOURCE_FROZEN',target_reads=0,query_reads=0,U_truth_reads=0,target_eligible=True,selected_rank=16)
    selection=parent/'source_selection.json';selection.write_text(json.dumps(selected))
    rt=dict(common,reuse_source_runtime=str(runtime_file),run_root=str(out))
    with pytest.raises(ValueError,match='ancestry'):reuse_source_checkpoint(dict(rt,model_seeds=[2]))
    selection.write_text(json.dumps(dict(selected,target_reads=1)))
    with pytest.raises(ValueError,match='source-only'):reuse_source_checkpoint(rt)
    selection.write_text(json.dumps(selected));reuse_source_checkpoint(rt)
    copied=PairedCCA.load(out/'source_cca.npz');original=PairedCCA.load(parent/'source_cca.npz')
    assert all(np.array_equal(a,b) for p,q in zip(original.parameters,copied.parameters) for a,b in zip(p,q))
    assert json.loads((out/'source_selection.json').read_text())['reuse_no_new_source_fit'] is True
