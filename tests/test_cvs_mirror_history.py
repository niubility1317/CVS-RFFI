import copy
import numpy as np
import pytest
import torch
from experiments.cvs_mirror_history.run import public_history,paired_fir,diagnose,DELAYS
from experiments.cvs_mirror_history.analyze import validate_arrays,records
from experiments.cvs_mirror_history.analyze import validate_completion,protect_outputs,csvwrite,OUTPUTS,RUN
from experiments.cvs_mirror_subspace_identity.model import build
from experiments.cvs_spectral_relation_identity.physics import causal_fir,multiply_pair

def test_public_history_is_repeatable_and_crop_normalized():
    a=public_history();assert torch.equal(a,public_history()) and a.shape==(30,2,272)
    power=a[...,16:].square().sum(1).mean(-1)
    assert torch.allclose(power[:24],torch.ones(24),atol=2e-7) and not power[24:].any()

@pytest.mark.parametrize('delay',DELAYS)
def test_zero_history_equals_original_fir_and_support_is_local(delay):
    h=public_history();z=paired_fir(h,delay,False);c=paired_fir(h,delay,True)
    assert torch.equal(z,causal_fir(h[...,16:],[(0,1.,0.),(delay,.23,.09)]))
    assert torch.equal(z[...,delay:],c[...,delay:])
    assert torch.any(z[18:24,:,:delay]!=c[18:24,:,:delay])

@pytest.mark.parametrize('delay',DELAYS)
def test_continuous_tones_match_analytic_complex_gain(delay):
    h=public_history().double();y=paired_fir(h,delay,True)[6:12]
    omega=.071+.021*torch.arange(1,7,dtype=torch.float64)
    phase=-omega*delay
    real=1+.23*phase.cos()-.09*phase.sin();imag=.23*phase.sin()+.09*phase.cos()
    expected=multiply_pair(h[6:12,:,16:],real[:,None],imag[:,None])
    assert (y-expected).abs().max()<5e-8

@pytest.mark.parametrize('variant',['mirror_energy','mirror_subspace'])
def test_full_packet_frame_and_frequency_recount(variant):
    torch.set_num_threads(2);torch.manual_seed(13)
    a=diagnose(build(variant));checks=validate_arrays(a,variant=='mirror_subspace')
    assert len(checks)==12
    p,f,t,e=records(a,variant,1)
    assert (len(p),len(f),len(t),len(e))==(180,5580,1260,90)
    assert all(r['raw_spectral_boundary_norm']==0 for r in t if r['frame']>0)
    bad=dict(a);bad['fp32_d16_continuous_raw_iq']=a['fp32_d16_continuous_raw_iq'].copy()
    bad['fp32_d16_continuous_raw_iq'][0,0,-1]+=1
    with pytest.raises(ValueError):validate_arrays(bad,variant=='mirror_subspace')

def complete_matrix():
    return dict(status='VERIFIED',run_id=RUN,target_access=False,public_packets_per_model=30,paired_cases=1440,
        rows=[dict(row_id=f'{v}-s{s}',variant=v,model_seed=s,status='VERIFIED',public_packets=30,cases=180,
            optimizer_steps=0,source_IQ_access=False,target_access=False,model_state_unchanged=True)
            for v in ('mirror_energy','mirror_subspace') for s in range(2026092701,2026092705)])

def test_fixed_matrix_rejects_duplicate_missing_and_scope_changes():
    good=complete_matrix();assert len(validate_completion(good))==8
    bad=copy.deepcopy(good);bad['rows']=[good['rows'][0]]*4+[good['rows'][4]]*4
    with pytest.raises(ValueError):validate_completion(bad)
    for key,value in [('run_id','another-run'),('target_access',True),('paired_cases',1439)]:
        bad=copy.deepcopy(good);bad[key]=value
        with pytest.raises(ValueError):validate_completion(bad)
    for key,value in [('row_id','../wrong'),('optimizer_steps',1),('source_IQ_access',True),('model_state_unchanged',False)]:
        bad=copy.deepcopy(good);bad['rows'][0][key]=value
        with pytest.raises(ValueError):validate_completion(bad)

@pytest.mark.parametrize('name',OUTPUTS)
def test_all_evidence_paths_are_preserved(tmp_path,name):
    p=tmp_path/name;p.write_bytes(b'original evidence')
    with pytest.raises(FileExistsError):protect_outputs(tmp_path)
    assert p.read_bytes()==b'original evidence'

@pytest.mark.parametrize('name',['packets.csv','frequency_pairs.csv.gz'])
def test_csv_exclusive_creation_prevents_racing_overwrite(tmp_path,name):
    p=tmp_path/name;p.write_bytes(b'original evidence')
    with pytest.raises(FileExistsError):csvwrite(p,[dict(value=1)])
    assert p.read_bytes()==b'original evidence'
