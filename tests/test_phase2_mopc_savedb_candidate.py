from experiments.pamf_phase2_practical import group2_savedb_candidate_diagnostic as candidate


def test_candidate_forwards_paper_semantics_and_diagnostic_clip(monkeypatch):
    seen=[]
    monkeypatch.setattr(candidate.cil,'register',lambda anchor,x,y,**kw: seen.append(kw) or 'state')
    prepare={'mopc_hr':'anchor'}
    assert candidate.register_candidate(prepare,'x','y',seed=17,methods=('mopc_hr',),clip=None)=={'mopc_hr':'state'}
    assert candidate.register_candidate(prepare,'x','y',seed=17,methods=('mopc_hr',),clip=1.)=={'mopc_hr':'state'}
    assert seen==[dict(seed=17,mopc_semantics='paper',mopc_max_grad_norm=None),
                  dict(seed=17,mopc_semantics='paper',mopc_max_grad_norm=1.)]
