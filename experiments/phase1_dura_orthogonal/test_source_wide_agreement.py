from experiments.phase1_dura_orthogonal import source_wide_agreement as wide


def test_fixed_candidate_matrix_is_passed_to_existing_probe(monkeypatch):
    seen=[]
    monkeypatch.setattr(wide.base,'main',lambda:seen.append(dict(wide.base.CANDIDATES)))
    wide.main()
    assert seen == [wide.CANDIDATES]
    assert len({row[0] for row in seen[0].values()}) == 3
