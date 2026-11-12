from experiments.pamf_phase2_practical.clean_comparison_plan import groups, tasks


def test_full_plan_and_explicit_method_paths():
    jobs = tasks()
    assert len(jobs) == 3 * 23 * 7 * 4
    assert len({(j['capsule'], j['seed']) for j in jobs}) == len(jobs)
    assert all(len(j['points']) == (16 if '/MIX' in j['capsule'] else 1) for j in jobs)
    spec = groups()
    assert spec['pamf']['paths'] == ['S0Q0', 'S1Q1']
    for name in ('native_da', 'native_cil', 'hybrid'):
        assert spec[name]['paths'] == ['S0Q0']
    assert spec['native_da']['methods'][-1] == 'tweak_m1'
    assert 'issl_fixed_v1_long' in spec['native_cil']['methods']
    assert len(spec['hybrid']['candidates']) == 18
    assert all('radionet' not in str(s).lower() and 'hu2024' not in str(s).lower() and 'ra_collab' not in str(s).lower()
               for s in spec.values())
