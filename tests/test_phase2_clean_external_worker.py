import pytest
from experiments.pamf_phase2_practical.clean_external_worker import validate


def test_only_bound_corrected_profiles_are_accepted():
    rt=dict(clean_root='/clean',data_root='/parent')
    base=dict(seed=2026092701,capsule='capsules/P01/MIX01/1-1',output='/out',points=[[5,20]])
    validate(rt,dict(base,profile='dadda_sda'),'native_da')
    validate(rt,dict(base,profile='tweak_m1'),'native_da')
    validate(rt,dict(base,profile='orthogonal'),'native_cil')
    validate(rt,dict(base,profile='issl_fixed_v1'),'native_cil')
    validate(rt,dict(base,profile='dadda_sda',methods=['orthogonal','lora']),'hybrid')
    validate(rt,dict(base,profile='tweak_m1',methods=['lora']),'hybrid')
    with pytest.raises(ValueError):validate(rt,dict(base,profile='tweak_m1',methods=['orthogonal']),'hybrid')
    for profile in ('radionet_ada_knn','feature_separation_hu2024'):
        with pytest.raises(ValueError):validate(rt,dict(base,profile=profile),'native_da')
    for profile in ('mopc_hr','issl_author_behavior'):
        with pytest.raises(ValueError):validate(rt,dict(base,profile=profile),'native_cil')
    with pytest.raises(ValueError):validate(rt,dict(base,profile='dadda_sda',methods=['mopc_hr']),'hybrid')
    with pytest.raises(ValueError):validate(rt,dict(base,profile='dadda_sda',scope='pilot_main'),'native_da')
    with pytest.raises(ValueError):validate(rt,dict(base,profile='dadda_sda',capsule='capsules/../MIX01/1-1'),'native_da')
